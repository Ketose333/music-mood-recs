"""Sync app.py's inline AUTO-SYNCED blocks from src/ (single source of truth).

The submission deliverable requires exactly one .py source file (과제 공지 —
보고서+ipynb+py 총 3개 파일), so app.py cannot `from src...` import. The model
and recommendation code lives for real in src/ and is exercised by tests/.
This script copies their current source text into app.py's marker blocks so
the two copies can never silently drift apart.

Run after editing any of the source files listed in BLOCKS.

Usage:
  python scripts/sync_standalone_app.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

APP_PY = Path("app.py")

# (src file, [top-level def/class/assignment names to inline, in source order])
BLOCKS: list[tuple[Path, list[str]]] = [
    (Path("src/models/cnn.py"), ["CNNConfig", "MoodCNN"]),
    (Path("src/recommend/similar.py"), ["extract_embeddings", "top_k_similar"]),
]


def _node_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _whole_module(src_path: Path) -> str:
    source = src_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    # Consecutive import statements are joined tightly (single newline); any
    # other statement (def/class/constant) starts a new chunk separated by a
    # blank-line gap, so inlined imports don't look inconsistently spaced
    # next to the cell's own hand-written import header.
    chunks: list[str] = []
    import_buf: list[str] = []
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            continue  # module docstring
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            continue  # internal cross-module ref — must already be defined by an earlier block
        decorators = getattr(node, "decorator_list", [])
        start_line = min([d.lineno for d in decorators] + [node.lineno])
        text = "".join(lines[start_line - 1 : node.end_lineno]).rstrip("\n")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_buf.append(text)
            continue
        if import_buf:
            chunks.append("\n".join(import_buf))
            import_buf = []
        chunks.append(text)
    if import_buf:
        chunks.append("\n".join(import_buf))
    return "\n\n\n".join(chunks) + "\n"


def _extract(src_path: Path, names: list[str]) -> str:
    if names == ["*"]:
        return _whole_module(src_path)
    source = src_path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    by_name = {_node_name(node): node for node in tree.body if _node_name(node) in names}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"{src_path}: could not find {missing}")
    segments = []
    for name in names:
        node = by_name[name]
        decorators = getattr(node, "decorator_list", [])
        start_line = min([d.lineno for d in decorators] + [node.lineno])
        segments.append("".join(lines[start_line - 1 : node.end_lineno]).rstrip("\n"))
    return "\n\n\n".join(segments) + "\n"


def main() -> int:
    text = APP_PY.read_text(encoding="utf-8")
    for src_path, names in BLOCKS:
        begin = f"# >>> AUTO-SYNCED from {src_path.as_posix()} (run scripts/sync_standalone_app.py) >>>"
        end = "# <<< AUTO-SYNCED <<<"
        pattern = re.compile(re.escape(begin) + r"\n.*?\n" + re.escape(end), re.DOTALL)
        if not pattern.search(text):
            raise SystemExit(f"Marker block not found in app.py for {src_path}")
        body = _extract(src_path, names)
        replacement = f"{begin}\n{body}{end}"
        text = pattern.sub(lambda _m, r=replacement: r, text, count=1)

    if re.search(r"^\s*from src(\.| )|^\s*import src(\.| |$)", text, re.MULTILINE):
        raise SystemExit("app.py must not import the src package — sync introduced a forbidden import")

    APP_PY.write_text(text, encoding="utf-8")
    print("app.py synced from src/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
