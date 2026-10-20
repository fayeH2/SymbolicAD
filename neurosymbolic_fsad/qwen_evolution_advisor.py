"""Local Qwen3-VL contextual feedback for symbolic rule evolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .qwen_rules import DEFAULT_MODEL_PATH, extract_json_object
from .symbolicdet_evolution import normalize_normality_ast, validate_normality_ast


class QwenEvolutionAdvisor:
    """Inject strictly validated AST suggestions into a running GA.

    The model sees only atom semantics and support-counterfactual fitness.  It
    never receives query images, query scores, defect names, or query labels.
    """

    def __init__(
        self,
        available_atoms: Sequence[str],
        atom_descriptions: Mapping[str, str],
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
        task_context: str = "MVTec transistor anomaly detection",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.available_atoms = tuple(str(name) for name in available_atoms)
        self.atom_descriptions = {
            name: str(atom_descriptions[name]) for name in self.available_atoms
        }
        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(self.model_path)
        target = device
        if target.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for Qwen feedback but CUDA is unavailable")
        dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=dtype, **kwargs
            )
        self.model = self.model.to(target).eval()
        self.device = target
        self.max_new_tokens = int(max_new_tokens)
        self.task_context = str(task_context).strip()

    def _prompt(self, generation: int, top: Sequence[Mapping[str, Any]]) -> str:
        schema = {
            "suggestions": [
                {
                    "op": "AND",
                    "args": [
                        {"op": "ATOM", "name": self.available_atoms[0]},
                        {
                            "op": "ATOM",
                            "name": self.available_atoms[-1],
                        },
                    ],
                }
            ]
        }
        return f"""
You guide an evolutionary search for a STATIC normality program for
{self.task_context}. The visual atoms were proposed by Qwen from
known-normal support images and calibrated only on those supports.

An ATOM value is a fuzzy normality truth: 1 means satisfied, 0 means violated.
The program returned here is a NORMALITY expression; the executor applies one
outer NOT to obtain anomaly. Product-fuzzy operators are:
AND=product, OR=1-product(1-x), NOT=1-x,
IMPLIES(a,b)=1-a*(1-b), XOR=a+b-2ab.

Available atoms and meanings:
{json.dumps(self.atom_descriptions, ensure_ascii=False, indent=2)}

Current generation: {generation}
Best support-counterfactual programs and fitness (NO query/test data):
{json.dumps(list(top), ensure_ascii=False, indent=2)}

Suggest 1 to 5 improved normality ASTs. Use only ATOM, NOT, AND, OR, IMPLIES,
XOR. Every chosen atom may occur at most once in an AST. Use only the available
atom names. Do not emit thresholds, arithmetic, code, prose, or unknown fields.
Prefer short expressions when scores are close.

NOT has the exact form {{"op":"NOT","arg":<one AST>}}; do not use an args
list for NOT. AND/OR/IMPLIES/XOR use an ``args`` list.

Return ONLY one JSON object shaped like:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

    def _generate(self, prompt: str) -> str:
        import torch

        prefix = '{"suggestions":['
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": prefix}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {
            name: value.to(model_device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        continuation = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return prefix + continuation

    def __call__(
        self, generation: int, top: Sequence[Mapping[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prompt = self._prompt(generation, top)
        raw = self._generate(prompt)
        parsed: Dict[str, Any]
        error = None
        suggestions: List[Dict[str, Any]] = []
        try:
            parsed = extract_json_object(raw)
            if set(parsed) != {"suggestions"} or not isinstance(
                parsed["suggestions"], list
            ):
                raise ValueError("response must contain only a suggestions list")
            invalid_suggestions = []
            for index, item in enumerate(parsed["suggestions"][:5]):
                if not isinstance(item, Mapping):
                    invalid_suggestions.append({"index": index, "error": "not an object"})
                    continue
                try:
                    candidate = normalize_normality_ast(item)
                    validate_normality_ast(candidate, self.available_atoms)
                    suggestions.append(candidate)
                except Exception as exc:
                    invalid_suggestions.append({
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except Exception as exc:  # audited fail-closed behavior
            error = f"{type(exc).__name__}: {exc}"
            invalid_suggestions = []
        return suggestions, {
            "model": self.model_path,
            "device": self.device,
            "prompt": prompt,
            "raw_output": raw,
            "parse_error": error,
            "invalid_suggestions": invalid_suggestions,
            "valid_suggestion_count": len(suggestions),
            "query_or_test_information_supplied": False,
        }

    def close(self) -> None:
        import gc
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["QwenEvolutionAdvisor"]
