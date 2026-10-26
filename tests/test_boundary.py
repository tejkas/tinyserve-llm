"""
The engine and model reach the KV cache only through engine/interface.py, so the contract between
them stays explicit and the cache can be swapped or ablated without touching engine code.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "tinyserve_llm"
GUARDED = ("engine", "model")
INTERFACE = PACKAGE / "engine" / "interface.py"
FORBIDDEN = ("tinyserve_llm.kvcache", "tinyserve_llm.offload")


def forbidden_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [n for n in names if n.startswith(FORBIDDEN)]
    return found


def test_checker_catches_both_import_forms():
    src = "import tinyserve_llm.kvcache.manager\nfrom tinyserve_llm.offload import x\nimport torch\n"
    assert forbidden_imports(src) == ["tinyserve_llm.kvcache.manager", "tinyserve_llm.offload"]


def test_engine_and_model_do_not_import_cache_internals():
    violations = {}
    for package in GUARDED:
        for path in (PACKAGE / package).rglob("*.py"):
            if path == INTERFACE:
                continue
            found = forbidden_imports(path.read_text())
            if found:
                violations[str(path.relative_to(PACKAGE))] = found
    assert not violations, f"go through engine/interface.py instead: {violations}"
