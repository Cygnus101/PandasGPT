"""
guard.py

Static guardrail for PandasGPT generated code.

Functions:
    validate_code(code_str: str, df_meta: dict, allowed_names: set[str] | None = None) -> dict
        Returns a dict: {'ok': bool, 'issues': list[str]}

The guard performs three fast static checks without executing the code:
1. **Syntax gate** – Using ast.parse.
2. **Safety visitor** – Blocks import statements, eval/exec, and dangerous attribute roots.
3. **Schema check** – Extracts df['col'] accesses and verifies them against df_meta['columns'].

Add the truth‑oracle executor later if runtime validation is required.
"""

from __future__ import annotations

import ast
import difflib
from typing import Dict, List, Set, Tuple
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base names that a generated snippet is allowed to reference without complaint
_BASE_ALLOWED_NAMES: Set[str] = {
    "df",        # the DataFrame itself
    "pd",        # pandas alias
    "np",        # numpy alias
    "plt",       # matplotlib.pyplot alias
}

# Attribute roots that are considered unsafe (prevent file/network/system ops)
_DISALLOWED_ATTR_PREFIXES: Tuple[str, ...] = (
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "builtins", "open", "eval", "exec", "__",
)

# -----------------------------------------------------------------------------
# AST Visitors
# -----------------------------------------------------------------------------

class SafeNodeVisitor(ast.NodeVisitor):
    """Walks an AST to collect column names and flag disallowed constructs."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.columns: Set[str] = set()

    # --- Import statements ---
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 (snake‑case enforced elsewhere)
        self.errors.append("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.errors.append("Import statements are not allowed")

    # --- eval/exec ---
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
            self.errors.append(f"Disallowed call to {node.func.id}()")
        self.generic_visit(node)

    # --- Dangerous attribute chains ---
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        root = self._root_name(node)
        if root and root.startswith(_DISALLOWED_ATTR_PREFIXES):
            self.errors.append(f"Disallowed attribute access '{root}.*'")
        self.generic_visit(node)

    # --- df["col"] accesses ---
    DATE_SLICE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")  # 'YYYY-MM' or 'YYYY-MM-DD'

    def visit_Subscript(self, node: ast.Subscript) -> None:
        DATE_SLICE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")  # 'YYYY-MM' or 'YYYY-MM-DD'
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "df"
            and isinstance(node.slice, (ast.Constant, ast.Str))
        ):
            col = node.slice.s if isinstance(node.slice, ast.Str) else node.slice.value

            # --- NEW: skip date-like literals; they’re index slices, not columns
            if isinstance(col, str) and DATE_SLICE_RE.match(col):
                # treat as a DatetimeIndex slice → nothing to record
                self.generic_visit(node)
                return

            if isinstance(col, str):
                self.columns.add(col)
        self.generic_visit(node)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _root_name(node: ast.Attribute) -> str | None:
        """Return the left‑most name in an Attribute chain (e.g., os.path.join → os)."""
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _check_columns(used: Set[str], known: Set[str]) -> List[str]:
    """Return list of error strings for columns that don't exist in df_meta."""
    issues: List[str] = []
    for col in used:
        if col not in known:
            suggestion = difflib.get_close_matches(col, known, n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            issues.append(f"Unknown column '{col}'.{hint}")
    return issues


def validate_code(code_str: str, df_meta: Dict[str, object], *, allowed_names: Set[str] | None = None) -> Dict[str, object]:
    """Fast static validation of a generated pandas snippet.

    Parameters
    ----------
    code_str : str
        The code snippet to validate (one‑liner or few lines).
    df_meta : dict
        Must supply at least {'columns': Iterable[str]}.
    allowed_names : set[str], optional
        Additional global names the snippet may reference (e.g., {'sns'}).

    Returns
    -------
    dict with keys:
        'ok'     → bool  (True if no issues)
        'issues' → list[str]  (empty if ok)
    """

    allowed_names = (_BASE_ALLOWED_NAMES | (allowed_names or set()))
    issues: List[str] = []

    # ------------------------------------------------------------------
    # 1. Syntax check
    # ------------------------------------------------------------------
    try:
        tree = ast.parse(code_str, mode="exec")
    except SyntaxError as exc:
        return {
            "ok": False,
            "issues": [f"SyntaxError: {exc.msg} (line {exc.lineno})"],
        }

    # ------------------------------------------------------------------
    # 2. Safety + column collection visitor
    # ------------------------------------------------------------------
    visitor = SafeNodeVisitor()
    visitor.visit(tree)
    issues.extend(visitor.errors)

    # ------------------------------------------------------------------
    # 3. Column‑name validation
    # ------------------------------------------------------------------
    known_cols: Set[str] = set(df_meta.get("columns", []))
    issues.extend(_check_columns(visitor.columns, known_cols))

    # ------------------------------------------------------------------
    # 4. Name validation (unknown globals)
    # ------------------------------------------------------------------
    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
            if isinstance(node.ctx, ast.Load) and node.id not in allowed_names and node.id not in visitor.columns:
                issues.append(f"Use of unknown variable '{node.id}'")

    NameCollector().visit(tree)

    return {"ok": not issues, "issues": issues}


# -----------------------------------------------------------------------------
# Optional CLI for quick manual testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    example_code = "df['Voltgae'].mean()"  # typo on purpose
    meta = {"columns": {"Voltage", "Global_active_power"}}
    result = validate_code(example_code, meta)
    from pprint import pprint
    pprint(result)
