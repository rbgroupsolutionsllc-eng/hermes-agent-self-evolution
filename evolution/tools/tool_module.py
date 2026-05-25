"""Inspect Hermes Agent tool schemas without importing or modifying them.

Strategy: parse tool source files with regex — never import the modules.
This avoids side effects (registry calls, dependency loads, env checks).

description_type classification:
  literal  — "description": "..." or "description": ("..." "...")
  variable — "description": SOME_CONSTANT,
  dynamic  — "description": some_function(), OR post-declaration override detected
  unknown  — pattern not recognized

Sprint 1.1: second-pass override detection for post-declaration mutations.
Sprint 2A: ToolModule (DSPy) + tool_fitness_metric for GEPA optimization.
"""

import re
from pathlib import Path
from typing import Optional


# ── Regex patterns for schema detection ──────────────────────────────────────

# Matches: SOMETHING_SCHEMA = {
_RE_SCHEMA_VAR = re.compile(r'^(\w+SCHEMA)\s*=\s*\{', re.MULTILINE)

# Matches: "name": "tool_name"  (tool-level name field inside schema)
_RE_NAME_FIELD = re.compile(r'"name"\s*:\s*"([^"]+)"')

# Matches the description field value — three forms:
#   literal string:   "description": "text" or "description": ("text" "more")
#   variable ref:     "description": UPPER_CASE_VAR,
#   dynamic call:     "description": _some_function(),
_RE_DESC_LITERAL = re.compile(
    r'"description"\s*:\s*'           # key
    r'(\('                            # opening paren (multiline concat)
    r'(?:[^()]*'                      # content inside parens
    r'(?:"[^"]*")'                    # at least one quoted string
    r'[^()]*)*\)'                     # close paren
    r'|"[^"]*")',                     # OR simple quoted string
    re.DOTALL,
)
_RE_DESC_VARIABLE = re.compile(r'"description"\s*:\s*([A-Z][A-Z0-9_]+)\s*[,\n]')
_RE_DESC_DYNAMIC = re.compile(r'"description"\s*:\s*(\w+\s*\()')

# ── Second-pass: post-declaration description override detection ──────────────

# Pattern A: SCHEMA_VAR["description"] = ...  or  SCHEMA_VAR['description'] = ...
_RE_OVERRIDE_BRACKET = re.compile(
    r"(\w+SCHEMA)\s*\[['\"]description['\"]\]\s*="
)

# Pattern B: SCHEMA_VAR.update({... 'description': ...})
_RE_OVERRIDE_UPDATE = re.compile(
    r"(\w+SCHEMA)\.update\s*\(\s*\{[^}]*['\"]description['\"]"
)

# Pattern C: registry.register( call containing name=tool_name AND dynamic_schema_overrides=
_RE_REGISTRY_DYNAMIC = re.compile(
    r"registry\.register\s*\(",
    re.DOTALL,
)


def _extract_string_value(raw: str) -> str:
    """Extract the actual text from a Python string literal or concat expression."""
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    parts = re.findall(r'"([^"]*)"', raw)
    return "".join(parts).replace("\\n", "\n").replace("\\t", "\t")


def _classify_description(schema_text: str) -> tuple[str, str]:
    """
    Given the text block of a schema dict, return (description_type, raw_value).
    Priority: dynamic > variable > literal > unknown
    """
    if _RE_DESC_DYNAMIC.search(schema_text):
        m = _RE_DESC_DYNAMIC.search(schema_text)
        return "dynamic", m.group(1).strip() if m else ""

    m = _RE_DESC_VARIABLE.search(schema_text)
    if m:
        return "variable", m.group(1).strip()

    m = _RE_DESC_LITERAL.search(schema_text)
    if m:
        return "literal", _extract_string_value(m.group(0).split(":", 1)[1].strip())

    return "unknown", ""


def _extract_schema_block(source: str, schema_start_pos: int) -> tuple[str, int]:
    """
    Extract the text of a schema dict starting at schema_start_pos.
    Returns (block_text, end_pos) where end_pos is the index AFTER the closing '}'.
    Uses brace-counting parser — no approximate string search.
    """
    depth = 0
    in_string = False
    escape = False
    i = source.find("{", schema_start_pos)
    if i == -1:
        return "", schema_start_pos

    start = i
    for j in range(i, min(i + 20_000, len(source))):
        ch = source[j]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch in ('"', "'") and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : j + 1], j + 1
    return source[start:], len(source)


def _detect_runtime_override(
    source: str,
    schema_var: str,
    tool_name: str,
) -> tuple[bool, str]:
    """
    Second-pass: detect post-declaration description overrides.

    Patterns (checked in order):
    A. schema_var['description'] = ...   (direct mutation)
    B. schema_var.update({'description': ...})
    C. registry.register() block with name=tool_name AND dynamic_schema_overrides=

    Returns (has_override, pattern_description).
    """
    p_a = re.compile(rf"{re.escape(schema_var)}\s*\[['\"]description['\"]\]\s*=")
    if p_a.search(source):
        return True, f'{schema_var}["description"] = ...'

    p_b = re.compile(rf"{re.escape(schema_var)}\.update\s*\(\s*\{{[^}}]*['\"]description['\"]")
    if p_b.search(source):
        return True, f'{schema_var}.update({{"description": ...}})'

    for reg_m in _RE_REGISTRY_DYNAMIC.finditer(source):
        call_text = source[reg_m.start() : reg_m.start() + 2000]
        depth = 0
        end_pos = len(call_text)
        for idx, ch in enumerate(call_text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_pos = idx + 1
                    break
        call_text = call_text[:end_pos]
        name_in_call = re.search(rf"""name\s*=\s*['"]{re.escape(tool_name)}['"]""", call_text)
        dynamic_in_call = re.search(r"dynamic_schema_overrides\s*=", call_text)
        if name_in_call and dynamic_in_call:
            return True, f"registry.register(name={tool_name!r}, dynamic_schema_overrides=...)"

    return False, ""


def find_tool(tool_name: str, hermes_agent_path: Path) -> Optional[dict]:
    """Find a tool schema by schema["name"] field.

    Scans hermes_agent_path/tools/*.py using regex — no imports, no side effects.

    Returns a dict with:
        name                     str   — schema["name"]
        description              str   — resolved description text (or <dynamic: ...>)
        parameters               None  — parsed on demand
        source_file              Path  — absolute path to the Python file
        schema_var               str   — e.g. "CLARIFY_SCHEMA"
        description_chars        int   — len(description)
        description_type         str   — literal | variable | dynamic | unknown
        static_description_chars int   — len of statically-captured desc text
        runtime_description_chars str  — "unknown" (not executed)
        runtime_override_detected bool
        override_pattern         str   — detected override pattern, or ""
        warning                  str   — human-readable warning if override detected
    """
    tools_dir = hermes_agent_path / "tools"
    if not tools_dir.exists():
        return None

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name in {
            "registry.py", "ansi_strip.py", "binary_extensions.py",
            "budget_config.py", "debug_helpers.py", "env_passthrough.py",
            "fuzzy_match.py", "interrupt.py", "lazy_deps.py",
            "file_state.py", "file_operations.py",
        }:
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for m in _RE_SCHEMA_VAR.finditer(source):
            schema_var = m.group(1)
            block, schema_block_end = _extract_schema_block(source, m.start())

            name_match = _RE_NAME_FIELD.search(block)
            if not name_match or name_match.group(1) != tool_name:
                continue

            desc_type, desc_raw = _classify_description(block)

            if desc_type == "variable":
                var_name = desc_raw
                var_match = re.search(
                    rf'^{re.escape(var_name)}\s*=\s*(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')',
                    source,
                    re.DOTALL | re.MULTILINE,
                )
                if var_match:
                    desc_raw = var_match.group(1).strip().strip('"""').strip("'''").strip('"').strip("'")
                else:
                    var_block_match = re.search(
                        rf'^{re.escape(var_name)}\s*=\s*"""(.*?)"""',
                        source,
                        re.DOTALL | re.MULTILINE,
                    )
                    if var_block_match:
                        desc_raw = var_block_match.group(1)

            description_text = (
                desc_raw
                if desc_type in ("literal", "variable")
                else f"<{desc_type}: {desc_raw}>"
            )

            # Save static length BEFORE any promotion to dynamic
            static_description_chars = len(desc_raw or "")

            has_override, override_pattern = _detect_runtime_override(source, schema_var, tool_name)
            if has_override:
                desc_type = "dynamic"
                description_text = f"<dynamic: {override_pattern}>"

            return {
                "name": tool_name,
                "description": description_text,
                "parameters": None,
                "source_file": py_file,
                "schema_var": schema_var,
                "description_chars": len(description_text),
                "description_type": desc_type,
                "static_description_chars": static_description_chars,
                "runtime_description_chars": "unknown",
                "runtime_override_detected": has_override,
                "override_pattern": override_pattern if has_override else "",
                "warning": (
                    "Static schema description may not match runtime description"
                    if has_override else ""
                ),
            }

    return None


def load_tool(tool_name: str, hermes_agent_path: Path) -> dict:
    """Load a tool schema. Raises if not found."""
    result = find_tool(tool_name, hermes_agent_path)
    if result is None:
        raise FileNotFoundError(
            f"Tool '{tool_name}' not found in {hermes_agent_path / 'tools'}. "
            f"Ensure a schema dict with name='{tool_name}' exists."
        )
    return result


# ── Sprint 2A: DSPy ToolModule + fitness metric ───────────────────────────────

try:
    import dspy as _dspy_module
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


def _require_dspy() -> None:
    if not _DSPY_AVAILABLE:
        raise ImportError("dspy is required for GEPA optimization. Install with: pip install dspy")


class ToolModule:
    """DSPy module wrapping a tool description as an optimizable parameter.

    The tool description becomes the signature instructions — GEPA optimizes these
    instructions to improve dispatch accuracy across the evaluation dataset.

    Usage:
        module = ToolModule("Use skill_view to read existing skill files.", tool_name="skill_view")
        pred = module(user_request="Show me the drywall estimator skill.")
        evolved_desc = module.get_evolved_description()
    """

    def __new__(cls, tool_description: str, tool_name: str = "skill_view"):
        _require_dspy()
        import dspy

        # Define the base signature class
        class _DispatchSig(dspy.Signature):
            user_request: str = dspy.InputField(
                desc="User request that needs to be routed to the correct tool"
            )
            competing_tools: str = dspy.InputField(
                desc=(
                    "Other tools that might be selected instead: "
                    "skill_manage (create/edit/delete/install skills), "
                    "session_search (recall past conversation context), "
                    "terminal (run shell commands, cat files)"
                )
            )
            selected_correctly: str = dspy.OutputField(
                desc=(
                    "Should this tool be selected for user_request? "
                    "Answer 'yes' or 'no', then explain the dispatch boundary. "
                    "For 'no' answers, name which tool from competing_tools should be used instead."
                )
            )

        # Tool description IS the signature instructions — GEPA optimizes this
        _Sig = _DispatchSig.with_instructions(tool_description)

        # Build the actual dspy.Module instance
        class _Inner(dspy.Module):
            def __init__(self, description: str, name: str):
                super().__init__()
                self._tool_name = name
                sig = _DispatchSig.with_instructions(description)
                self.predictor = dspy.ChainOfThought(sig)

            def get_evolved_description(self) -> str:
                """Extract the (possibly GEPA-evolved) tool description."""
                return self.predictor.predict.signature.instructions

            def forward(
                self,
                user_request: str,
                competing_tools: str = "skill_manage, session_search, terminal",
            ):
                return self.predictor(
                    user_request=user_request,
                    competing_tools=competing_tools,
                )

        return _Inner(tool_description, tool_name)


def tool_fitness_metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """Fitness metric for tool dispatch evaluation.

    Scoring breakdown (sums to 1.0 for correct cases):
      dispatch_correct (0.5)   — yes for positive cases, no for negative cases
      negative_handling (0.3)  — correct competing tool named in rejection reasoning
      reasoning_alignment (0.2)— reasoning contains expected_behavior keywords

    Strong false positive penalty: predicted=yes when expected=no → score 0.0.
    """
    output = getattr(prediction, "selected_correctly", "")
    if not output:
        return 0.0
    output_lower = output.lower().strip()

    expected_positive = getattr(example, "expected_positive", True)
    predicted_yes = output_lower.startswith("yes")

    # Hard false positive penalty
    if not expected_positive and predicted_yes:
        return 0.0

    dispatch_correct = (predicted_yes == expected_positive)
    dispatch_score = 0.5 if dispatch_correct else 0.0

    # Negative handling (0.3)
    negative_score = 0.0
    if not expected_positive and not predicted_yes:
        correct_tool = getattr(example, "correct_tool", "").lower().replace("_", " ")
        # Accept underscore and space forms: "skill_manage" → "skill manage"
        if correct_tool and (
            correct_tool in output_lower or correct_tool.replace(" ", "_") in output_lower
        ):
            negative_score = 0.3
        else:
            negative_score = 0.15  # Partial: correct rejection, wrong competing tool named
    elif expected_positive and predicted_yes:
        negative_score = 0.3  # Positive cases: full credit if dispatch correct

    # Reasoning alignment (0.2)
    expected = getattr(example, "expected_behavior", "").lower()
    reasoning_score = 0.0
    if expected and dispatch_correct:
        stopwords = {
            "the", "a", "an", "is", "to", "in", "of", "and", "or", "not",
            "for", "it", "be", "this", "use", "not.", "skill.", "tool.",
        }
        keywords = set(expected.split()) - stopwords
        if keywords:
            matches = sum(1 for k in keywords if k in output_lower)
            reasoning_score = 0.2 * min(1.0, matches / len(keywords))
        else:
            reasoning_score = 0.1

    return min(1.0, dispatch_score + negative_score + reasoning_score)


# ── Sprint 2I: Variable-aware write pipeline ──────────────────────────────────

import shutil as _shutil
import datetime as _datetime


def _find_variable_definition(source: str, var_name: str) -> tuple[int, int, str]:
    """Locate the assignment block for a module-level string variable.

    Searches for triple-quote (\"\"\" or '''), then single-line forms.
    Returns (start_pos, end_pos, quote_style) where end_pos is exclusive.
    Raises ValueError if not found, multiple definitions detected, or block cannot be delimited.
    """
    triple_dq = re.compile(
        rf'^{re.escape(var_name)}\s*=\s*(""")(.*?)\1[ \t]*\n',
        re.MULTILINE | re.DOTALL,
    )
    triple_sq = re.compile(
        rf"^{re.escape(var_name)}\s*=\s*(''')(.*?)\1[ \t]*\n",
        re.MULTILINE | re.DOTALL,
    )
    single_line = re.compile(
        rf'^{re.escape(var_name)}\s*=\s*(["\']).*?\1[ \t]*\n',
        re.MULTILINE,
    )

    all_matches = []
    for pattern, style in [
        (triple_dq, '"""'),
        (triple_sq, "'''"),
        (single_line, "single"),
    ]:
        for m in pattern.finditer(source):
            all_matches.append((m.start(), m.end(), style))

    if not all_matches:
        raise ValueError(f"Cannot locate definition block for variable {var_name!r}")

    # Deduplicate by start position (triple patterns may overlap with single-line)
    by_start: dict[int, tuple[int, int, str]] = {}
    for start, end, style in all_matches:
        if start not in by_start:
            by_start[start] = (start, end, style)
        else:
            # Prefer triple-quote over single-line at the same position
            existing = by_start[start]
            if existing[2] == "single" and style != "single":
                by_start[start] = (start, end, style)

    unique = sorted(by_start.values())
    if len(unique) > 1:
        raise ValueError(
            f"Multiple definition blocks found for {var_name!r} at positions "
            + ", ".join(str(s) for s, _, _ in unique)
        )

    start, end, style = unique[0]
    if end <= start:
        raise ValueError(f"Could not delimit definition block for {var_name!r}")
    return start, end, style


def _verify_variable_exclusive(
    source_file: Path,
    var_name: str,
    hermes_agent_path: Path,
) -> tuple[bool, str]:
    """Verify the variable is used exclusively in source_file with exactly 2 references.

    Returns (True, "exclusive") if:
      - var_name appears exactly 2 times in source_file (definition + schema reference)
      - var_name appears 0 times in any other *.py file in hermes_agent_path/tools/

    Returns (False, reason) otherwise — caller must BLOCK the write.
    """
    try:
        source = source_file.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"Cannot read source file: {e}"

    occurrences = re.findall(rf'\b{re.escape(var_name)}\b', source)
    if len(occurrences) != 2:
        return False, (
            f"Expected exactly 2 references in source, found {len(occurrences)}. "
            "Write blocked."
        )

    tools_dir = hermes_agent_path / "tools"
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.resolve() == source_file.resolve():
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(rf'\b{re.escape(var_name)}\b', content):
            return False, f"Variable also referenced in {py_file.name}. Write blocked."

    return True, "exclusive"


def _render_python_string_assignment(var_name: str, new_desc: str, quote_style: str) -> str:
    """Render a valid Python string assignment for the given variable.

    For triple-quote styles, preserves the original quote style.
    For single-line (quote_style == "single"), uses double-quotes.
    Always produces a newline-terminated assignment.
    Raises ValueError if new_desc contains non-ASCII or control characters.
    """
    if not all(ord(c) <= 127 for c in new_desc):
        raise ValueError("Non-ASCII characters in new description — blocked.")
    if not all(ord(c) >= 32 or c in ('\n', '\t') for c in new_desc):
        raise ValueError("Control characters in new description — blocked.")

    if quote_style in ('"""', "'''"):
        # Keep as single-line assignment regardless of original triple-quote form
        safe = new_desc.replace('\\', '\\\\').replace('"', '\\"')
        return f'{var_name} = "{safe}"\n'
    else:
        safe = new_desc.replace('\\', '\\\\').replace('"', '\\"')
        return f'{var_name} = "{safe}"\n'


def _write_variable_description_to_text(
    source: str,
    var_name: str,
    new_desc: str,
) -> str:
    """Replace the variable assignment block in source with a new single-line assignment.

    Does NOT touch any file on disk — operates on strings only.
    Returns the modified source string.
    Raises ValueError if the variable block cannot be located or content validation fails.
    """
    start, end, quote_style = _find_variable_definition(source, var_name)
    new_assignment = _render_python_string_assignment(var_name, new_desc, quote_style)
    return source[:start] + new_assignment + source[end:]


def save_tool_description(
    tool_name: str,
    hermes_agent_path: Path,
    new_description: str,
    output_only: bool = True,
) -> dict:
    """Write a new description for a tool schema in the Hermes agent source.

    Supports description_type == "variable" only.
    output_only=True  — preview/diff only; no files written.
    output_only=False — writes to the real source file after all gates pass.
                        Creates a timestamped backup before writing.
                        Reverts from backup on py_compile failure or forbidden diff.

    Returns a result dict with keys:
        success           bool
        description_type  str
        old_description   str
        new_description   str
        diff_lines        list[str]
        block_reason      str   (non-empty when blocked before write)
        error             str   (non-empty on unexpected failure or post-write revert)
        backup_path       str   (non-empty when backup was created)
        wrote_file        bool  (True only when file was actually written)
        reverted          bool  (True if post-write revert was triggered)
    """
    import difflib as _difflib
    import py_compile as _py_compile

    result: dict = {
        "success": False,
        "description_type": "",
        "old_description": "",
        "new_description": new_description,
        "diff_lines": [],
        "block_reason": "",
        "error": "",
        "backup_path": "",
        "wrote_file": False,
        "reverted": False,
    }

    # ── Gate: candidate length
    if len(new_description) > 500:
        result["block_reason"] = f"Candidate {len(new_description)} chars > 500 limit. Blocked."
        return result

    # ── Gate: candidate ASCII + control chars
    if not all(ord(c) <= 127 for c in new_description):
        result["block_reason"] = "Non-ASCII characters in candidate. Blocked."
        return result
    if not all(ord(c) >= 32 or c in ('\n', '\t') for c in new_description):
        result["block_reason"] = "Control characters in candidate. Blocked."
        return result

    # ── Load tool metadata
    tool = find_tool(tool_name, hermes_agent_path)
    if tool is None:
        result["error"] = f"Tool '{tool_name}' not found."
        return result

    result["description_type"] = tool["description_type"]

    if tool["description_type"] == "literal":
        result["block_reason"] = (
            "description_type=literal is not writable by this pipeline. "
            "Use direct file edit for literal descriptions."
        )
        return result

    if tool["description_type"] != "variable":
        result["block_reason"] = (
            f"description_type={tool['description_type']!r} is not writable by this pipeline. "
            "Only 'variable' type is supported."
        )
        return result

    source_file: Path = tool["source_file"]
    schema_var: str = tool["schema_var"]

    # ── Read source
    try:
        source = source_file.read_text(encoding="utf-8")
    except OSError as e:
        result["error"] = f"Cannot read source file: {e}"
        return result

    # ── Re-extract schema block to resolve variable name
    schema_m = re.search(rf'^{re.escape(schema_var)}\s*=\s*\{{', source, re.MULTILINE)
    if not schema_m:
        result["error"] = f"Cannot locate schema variable {schema_var!r} in source."
        return result
    schema_block, _ = _extract_schema_block(source, schema_m.start())
    desc_type, var_name = _classify_description(schema_block)
    if desc_type != "variable":
        result["error"] = f"Re-classification returned {desc_type!r}, expected 'variable'."
        return result

    # ── Exclusive-use verification (BLOCK before any write)
    ok, reason = _verify_variable_exclusive(source_file, var_name, hermes_agent_path)
    if not ok:
        result["block_reason"] = reason
        return result

    # ── Locate variable definition (BLOCK if not found)
    try:
        def_start, def_end, quote_style = _find_variable_definition(source, var_name)
    except ValueError as e:
        result["block_reason"] = str(e)
        return result

    # ── Build new source in-memory
    try:
        new_source = _write_variable_description_to_text(source, var_name, new_description)
    except ValueError as e:
        result["block_reason"] = str(e)
        return result

    # ── Compute diff lines
    old_lines = source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)
    diff_lines = list(_difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{source_file.name}",
        tofile=f"b/{source_file.name}",
        n=3,
    ))
    result["diff_lines"] = diff_lines

    # ── Compute old_description from source block
    old_block = source[def_start:def_end]
    result["old_description"] = (
        old_block.strip()
        .removeprefix(f'{var_name} = """').removesuffix('"""')
        .removeprefix(f"{var_name} = '''").removesuffix("'''")
        .removeprefix(f'{var_name} = "').removesuffix('"')
        .removeprefix(f"{var_name} = '").removesuffix("'")
        .strip()
    )

    # ── Pre-write diff gate: forbidden schema structure changes
    _forbidden_in_diff = ['"name"', '"parameters"', schema_var]
    changed_lines = [
        l for l in diff_lines
        if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))
    ]
    for line in changed_lines:
        for forbidden in _forbidden_in_diff:
            if forbidden in line and var_name not in line:
                result["block_reason"] = (
                    f"Diff contains unexpected change ({forbidden!r} outside {var_name}). Blocked."
                )
                return result

    # ── output_only=True: preview only, no write
    if output_only:
        result["success"] = True
        return result

    # ── output_only=False: actual write path (variable type only — all gates passed above)

    # Backup (non-negotiable)
    backup_ts = _datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = source_file.with_name(
        source_file.stem + f".py.backup_sprint2i_b_{backup_ts}"
    )
    _shutil.copy2(source_file, backup_path)
    if backup_path.stat().st_size != source_file.stat().st_size:
        result["error"] = "Backup size mismatch — aborting before write."
        backup_path.unlink(missing_ok=True)
        return result
    result["backup_path"] = str(backup_path)

    # Write to real file
    source_file.write_text(new_source, encoding="utf-8")
    result["wrote_file"] = True

    # py_compile — revert on failure
    try:
        _py_compile.compile(str(source_file), doraise=True)
    except _py_compile.PyCompileError as e:
        _shutil.copy2(backup_path, source_file)
        result["error"] = f"py_compile failed — reverted from backup: {e}"
        result["wrote_file"] = False
        result["reverted"] = True
        return result

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# Sprint 3C — Function-body write pipeline
# ---------------------------------------------------------------------------

def _find_function_definition(source: str, function_name: str) -> tuple[int, int]:
    """Locate a module-level function by name and return (start, end) char positions.

    start = position of the 'd' in 'def function_name'
    end   = position of the first char of the NEXT top-level def/class/decorator
            (i.e. source[start:end] is the full function text + trailing blank lines)

    Raises ValueError if the function is not found or found more than once.
    """
    import re as _re
    func_pattern = _re.compile(
        rf"^def {_re.escape(function_name)}\b",
        _re.MULTILINE,
    )
    matches = list(func_pattern.finditer(source))
    if len(matches) == 0:
        raise ValueError(f"Function '{function_name}' not found in source")
    if len(matches) > 1:
        raise ValueError(
            f"Function '{function_name}' found {len(matches)} times — ambiguous"
        )

    start = matches[0].start()

    # Find next top-level symbol (def / class / decorator) after `start`
    next_toplevel = _re.compile(r"^(?:def |class |@[a-zA-Z_])", _re.MULTILINE)
    m = next_toplevel.search(source, start + 1)
    end = m.start() if m is not None else len(source)

    return start, end


def _render_delegate_description_function(candidate_template: str) -> str:
    """Render a replacement Python function body for _build_top_level_description().

    Validates that candidate_template contains {max_children} and {nesting_clause}
    placeholders, then returns a complete, syntactically valid Python function
    that:
      - reads max_children, max_depth, orchestrator_on from the same config
        reader calls as the original (_get_max_concurrent_children, etc.)
      - implements 3 nesting_clause branches:
          orchestrator disabled  → "Orchestrator disabled."
          depth <= 1             → "Nesting OFF (depth={max_depth})."
          depth > 1              → "Nesting ON (depth={max_depth})."
      - returns the candidate_B text with {max_children} and {nesting_clause}
        injected as f-string values
    Raises ValueError if required placeholders are missing from candidate_template.
    """
    required = ["{max_children}", "{nesting_clause}"]
    missing = [p for p in required if p not in candidate_template]
    if missing:
        raise ValueError(
            f"candidate_template missing required placeholders: {missing}"
        )

    # Split the candidate_B text around the two dynamic placeholders so we can
    # emit them as f-string interpolations.
    # Expected tail: "{max_children} parallel max. {nesting_clause} Pass context..."
    # We render the whole return as a single f-string.

    # Escape backslashes and double-quotes that would break the f-string literal.
    # The candidate_B text uses → (U+2192) which is fine in UTF-8 source.
    # We emit the string parts as escaped Python string concatenation.

    # Build the rendered function as a text block.
    fn = '''\
def _build_top_level_description() -> str:
    """Compose the delegate_task tool description with current runtime limits."""
    try:
        max_children = _get_max_concurrent_children()
    except Exception:
        max_children = _DEFAULT_MAX_CONCURRENT_CHILDREN
    try:
        max_depth = _get_max_spawn_depth()
    except Exception:
        max_depth = MAX_DEPTH
    try:
        orchestrator_on = _get_orchestrator_enabled()
    except Exception:
        orchestrator_on = True

    if not orchestrator_on:
        nesting_clause = "Orchestrator disabled."
    elif max_depth <= 1:
        nesting_clause = f"Nesting OFF (depth={max_depth})."
    else:
        nesting_clause = f"Nesting ON (depth={max_depth})."

    return (
        "Spawn subagents for isolated reasoning, research, or parallel work. "
        "Only final summaries return to your context.\\n\\n"
        "WHEN TO USE: analysis flooding context; code review/debugging; "
        "research synthesis; parallel independent subtasks.\\n\\n"
        "WHEN NOT TO USE:\\n"
        "- Shell/terminal commands -> terminal\\n"
        "- Save/recall/forget preferences -> memory\\n"
        "- View/create/edit/install skills -> skill tools\\n"
        "- Search past conversation -> session_search\\n"
        "- Ask user for missing input -> clarify\\n"
        "- Destructive ops, secrets, prod writes, unreviewed pushes -> BLOCKED\\n\\n"
        f"{max_children} parallel max. {nesting_clause} "
        "Pass context explicitly. Verify side-effects - summaries are SELF-REPORTS."
    )
'''
    return fn


def _write_function_body_to_text(
    source: str,
    function_name: str,
    new_function_text: str,
) -> str:
    """Replace a module-level function's complete block with new_function_text.

    Locates function_name via _find_function_definition(), replaces
    source[start:end] with new_function_text, and returns the new source string.
    The caller is responsible for py_compile validation before writing to disk.
    """
    start, end = _find_function_definition(source, function_name)
    # Preserve trailing blank lines between functions: new_function_text must
    # end with a single newline; we keep whatever blank lines were in source[end-?:end].
    # Since source[start:end] includes trailing blank lines up to the next symbol,
    # and new_function_text ends with '\n', we append '\n\n' to match PEP-8 spacing.
    if not new_function_text.endswith("\n"):
        new_function_text += "\n"
    return source[:start] + new_function_text + "\n\n" + source[end:]
