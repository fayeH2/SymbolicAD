# Qwen-guided relational normality rules for FSAD


This repository is an isolated neuro-symbolic layer around INP-Former.  It does
not modify the baseline repository.  Local
`/home/hjh/AgentModels/Qwen3-VL-8B-Thinking` is used once, offline, to induce a
normal scene program from few-shot normal images.  Test-time inference loads
only the compiled JSON rule library and deterministic visual grounder.
