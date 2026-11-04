import json

import pytest

from symbolicAD.static_program import (
    STATIC_PROGRAM_VERSION,
    StaticProgramPaths,
    load_manifest,
)


def _paths(tmp_path):
    return StaticProgramPaths(
        root=tmp_path,
        rules=tmp_path / "rules.json",
        support_memory=tmp_path / "support_memory.npz",
        symbolicdet=tmp_path / "symbolicdet.npz",
        manifest=tmp_path / "manifest.json",
    )


def test_static_manifest_fails_closed_without_runtime_contract(tmp_path):
    paths = _paths(tmp_path)
    paths.manifest.write_text(
        json.dumps({"version": STATIC_PROGRAM_VERSION}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="deployment contract"):
        load_manifest(paths)


def test_static_manifest_accepts_llm_free_contract(tmp_path):
    paths = _paths(tmp_path)
    payload = {
        "version": STATIC_PROGRAM_VERSION,
        "runtime_contract": {
            "llm_allowed": False,
            "query_set_adaptation": False,
        },
    }
    paths.manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert load_manifest(paths) == payload
