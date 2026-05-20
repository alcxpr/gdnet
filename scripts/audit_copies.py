"""
Thank you Claude...
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

COPY_METHODS = {
    "contiguous": "layout copy if not already contiguous",
    "to": "dtype/device cast -- copy unless same dtype+device",
    "clone": "explicit copy",
    "float": "cast to fp32 -- copy",
    "bfloat16": "cast to bf16 -- copy",
    "half": "cast to fp16 -- copy",
    "cpu": "device copy",
    "cuda": "device copy",
}

DETACH_METHODS = {"detach"}

COPY_TORCH_FUNCS = {
    "pad": "F.pad -- allocates padded copy",
    "cat": "torch.cat -- allocates concatenated tensor",
    "stack": "torch.stack -- allocates stacked tensor",
}

OOP_TO_INPLACE = {
    "add": "add_",
    "mul": "mul_",
    "div": "div_",
    "sub": "sub_",
}

CHAIN_PATTERNS = [
    (["permute", "contiguous"], "permuted tensor forced to contiguous layout -- copy"),
    (
        ["contiguous", "view"],
        "contiguous().view() -- consider .reshape() to avoid copy when possible",
    ),
    (
        ["contiguous", "reshape"],
        "contiguous().reshape() -- the contiguous() is redundant before reshape",
    ),
]


@dataclass
class Finding:
    path: str
    line: int
    col: int
    category: str
    detail: str
    snippet: str = ""

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}:{self.col}"
        return f"  [{self.category}] {loc}\n    {self.detail}\n    {self.snippet}"


class CopyAuditor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str], path: str) -> None:
        self.lines = source_lines
        self.path = path
        self.findings: list[Finding] = []

    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", None)
        if line is None:
            return ""
        return self.lines[line - 1].rstrip()

    def _add(self, node: ast.AST, category: str, detail: str) -> None:
        self.findings.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                category=category,
                detail=detail,
                snippet=self._snippet(node),
            )
        )

    @staticmethod
    def _method_chain(node: ast.expr) -> list[str]:
        chain: list[str] = []
        cur = node
        while isinstance(cur, ast.Call):
            if isinstance(cur.func, ast.Attribute):
                chain.append(cur.func.attr)
                cur = cur.func.value
            else:
                break
        return chain

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            if name in COPY_METHODS:
                self._add(node, "copy", f".{name}() - {COPY_METHODS[name]}")
            elif name in DETACH_METHODS:
                self._add(
                    node, "detach", ".detach() - graph boundary, audit if grad needed"
                )
            elif name in OOP_TO_INPLACE:
                self._add(
                    node,
                    "inplace?",
                    f".{name}() has in-place form .{OOP_TO_INPLACE[name]}() - safe if LHS not reused",
                )
            chain = self._method_chain(node)
            for pattern, msg in CHAIN_PATTERNS:
                rev = list(reversed(pattern))
                if chain[: len(rev)] == rev:
                    self._add(node, "chain", msg)
                    break
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:  # noqa: N802
        self.generic_visit(node)


class FuncCallAuditor(ast.NodeVisitor):
    def __init__(
        self, source_lines: list[str], path: str, findings: list[Finding]
    ) -> None:
        self.lines = source_lines
        self.path = path
        self.findings = findings

    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", None)
        if line is None:
            return ""
        return self.lines[line - 1].rstrip()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in COPY_TORCH_FUNCS:
            self.findings.append(
                Finding(
                    path=self.path,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    category="alloc",
                    detail=f"{func.attr}() -- {COPY_TORCH_FUNCS[func.attr]}",
                    snippet=self._snippet(node),
                )
            )
        self.generic_visit(node)


def audit_file(path: Path) -> list[Finding]:
    src = path.read_text()
    lines = src.splitlines()
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        print(f"  SyntaxError in {path}: {e}", file=sys.stderr)
        return []

    v1 = CopyAuditor(lines, str(path))
    v1.visit(tree)
    v2 = FuncCallAuditor(lines, str(path), v1.findings)
    v2.visit(tree)

    seen: set[tuple] = set()
    deduped: list[Finding] = []
    for f in v1.findings:
        key = (f.line, f.col, f.category)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    return sorted(deduped, key=lambda f: f.line)


CATEGORY_ORDER = ["copy", "alloc", "chain", "inplace?", "detach"]

CATEGORY_DESC = {
    "copy": "Copies (dtype cast / contiguous / clone)",
    "alloc": "Allocations (pad / cat / stack)",
    "chain": "Chained patterns (permute+contiguous, contiguous+view)",
    "inplace?": "Potential in-place ops",
    "detach": "Graph boundaries (detach)",
}


def report(findings: list[Finding], show_detach: bool = False) -> None:
    by_file: dict[str, list[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.path, []).append(f)

    total = 0
    for path, fs in sorted(by_file.items()):
        by_cat: dict[str, list[Finding]] = {}
        for f in fs:
            if f.category == "detach" and not show_detach:
                continue
            by_cat.setdefault(f.category, []).append(f)

        if not by_cat:
            continue

        rel = (
            Path(path).relative_to(Path.cwd())
            if Path(path).is_relative_to(Path.cwd())
            else Path(path)
        )
        print(f"\n{'=' * 70}")
        print(f"  {rel}  ({sum(len(v) for v in by_cat.values())} findings)")
        print(f"{'=' * 70}")

        for cat in CATEGORY_ORDER:
            items = by_cat.get(cat, [])
            if not items:
                continue
            print(f"\n  -- {CATEGORY_DESC.get(cat, cat)} ({len(items)}) --")
            for f in items:
                print(f"  L{f.line:4d}  {f.detail}")
                print(f"         {f.snippet.strip()}")
            total += len(items)

    print(f"\n{'=' * 70}")
    print(f"  Total findings: {total}  (run with --detach to include detach() calls)")
    print(f"{'=' * 70}\n")


DEFAULT_PATHS = [
    "gdnet/kernel/gated_causal_depthwise_conv/function.py",
    "gdnet/kernel/gated_causal_depthwise_conv/gate_norm.py",
    "gdnet/kernel/gated_causal_depthwise_conv/conv.py",
    "gdnet/kernel/fused_mem_read/function.py",
    "gdnet/kernel/fused_mem_read/kernel.py",
    "gdnet/layer.py",
    "gdnet/model.py",
    "gdnet/loss.py",
    "gdnet/memory.py",
    "gdnet/operators.py",
    "gdnet/utils/fp8.py",
    "gdnet/utils/sp.py",
]


def main() -> None:
    show_detach = "--detach" in sys.argv
    paths_arg = [a for a in sys.argv[1:] if not a.startswith("--")]

    root = Path(__file__).parent.parent
    if paths_arg:
        paths = [Path(p) for p in paths_arg]
    else:
        paths = [root / p for p in DEFAULT_PATHS]

    all_findings: list[Finding] = []
    for p in paths:
        if not p.exists():
            print(f"  skip (not found): {p}", file=sys.stderr)
            continue
        all_findings.extend(audit_file(p))

    report(all_findings, show_detach=show_detach)


if __name__ == "__main__":
    main()
