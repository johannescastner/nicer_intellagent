"""PR-O — Bug O: ``KeyError('output')`` on paused tool_calls in
``DialogGraph.chat_bot_node``.

Background:
  ``simulator/agents_graphs/dialog_graph.py`` builds a per-turn
  ``all_tool_calls`` dict by iterating messages after the last human
  message. AIMessage(tool_calls=[...]) entries register tool_call
  dicts keyed by id; ToolMessage entries set the ``output`` key on
  the matching tool_call. The loop then calls
  ``self.memory.insert_tool(..., v['output'])`` for every collected
  tool_call.

  Failure mode: when an AIMessage emits a tool_call (e.g.
  ``ask_human``) that calls ``interrupt()``, the LangGraph pause
  prevents the matching ToolMessage from being emitted. The tool_call
  dict in ``all_tool_calls`` therefore has NO ``output`` key. The
  direct ``v['output']`` subscript at the insert-loop raises
  ``KeyError('output')``. ``simulator/utils/parallelism.py:34/76``
  catches it as ``Error in chain invoke: 'output'`` and retries; on
  retry exhaustion the scenario is dropped.

  The downstream sink's typed signature is the principled fix-shape
  signal:

      SqliteSaver.insert_tool(self, thread_id: str, tool_name: str,
                              input: Optional[str],
                              output: Optional[str])

  The ``Optional[str]`` on ``output`` is the explicit invitation:
  ``None`` is a valid value. SQLite stores ``None`` as NULL, which
  preserves the "tool_call is pending — no output yet" semantic for
  any future reader of the Tools table.

  Fix: replace the direct subscript ``v['output']`` with
  ``v.get('output')`` (no default — returns ``None`` for missing key,
  matching the typed contract).

  Cross-bundle alignment: this is the conceptual mirror of baby-NICER
  PR1's paused-interrupt orphan gate. Same shape (tool_call without
  matching ToolMessage from ``interrupt()``), different repo,
  different fix point.

These sentinels lock the structural invariant via AST walks — no
regex, no source-text matching. Pure ``ast.parse`` + ``ast.walk``.

Run from intellagent root:

    pytest tests/test_pr_o_paused_tool_call_atomicity.py -v
"""
from __future__ import annotations

import ast
import inspect


# ============================================================================
# Helpers — AST walk utilities (no regex, structural matching only)
# ============================================================================


def _chat_bot_node_funcdef() -> ast.AST | None:
    """Locate the nested ``chat_bot_node`` function definition inside
    ``DialogGraph.get_chatbot_node`` (or its current refactored
    location). Returns the FunctionDef node, or None if not found.
    """
    from simulator.agents_graphs import dialog_graph

    source = inspect.getsource(dialog_graph)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == "chat_bot_node":
            return node
    return None


def _is_output_subscript(node: ast.AST) -> bool:
    """True iff ``node`` is a ``Subscript`` of the form
    ``<Name>['output']`` — direct dict subscript that raises
    KeyError when the key is missing. The Name's id can be anything
    (``v``, ``tc``, ``tool_call``, etc.) because the data shape is
    invariant: each is a tool_call dict where ``output`` is only
    present after a matching ToolMessage.
    """
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name):
        return False
    slice_node = node.slice
    if not (
        isinstance(slice_node, ast.Constant)
        and isinstance(slice_node.value, str)
        and slice_node.value == "output"
    ):
        return False
    return True


def _is_output_get_call(node: ast.AST) -> bool:
    """True iff ``node`` is a call ``<Name>.get('output', ...)`` —
    the principled defensive accessor that returns ``None`` (or the
    default) for missing keys."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Name)
    ):
        return False
    if not node.args:
        return False
    first = node.args[0]
    return (
        isinstance(first, ast.Constant)
        and isinstance(first.value, str)
        and first.value == "output"
    )


# ============================================================================
# Sentinel 1: NO direct ``X['output']`` subscript in chat_bot_node.
# ============================================================================


def test_no_direct_output_subscript_in_chat_bot_node():
    """The function body must not contain any ``<Name>['output']``
    subscript on a tool_call dict. The ``output`` key is OPTIONAL
    (only set when a ToolMessage matches the tool_call); direct
    subscript raises ``KeyError`` on paused interrupts.

    Locks the structural invariant so a refactor cannot silently
    re-introduce the failure mode.
    """
    func = _chat_bot_node_funcdef()
    assert func is not None, (
        "Could not locate chat_bot_node FunctionDef in "
        "simulator/agents_graphs/dialog_graph.py — has the function "
        "been renamed or restructured?"
    )

    offending_lines: list[int] = []
    for node in ast.walk(func):
        if _is_output_subscript(node):
            offending_lines.append(node.lineno)

    assert not offending_lines, (
        "chat_bot_node contains direct ``X['output']`` subscript(s) at "
        f"lines {offending_lines}. The ``output`` key is OPTIONAL on "
        "tool_call dicts (set only when a matching ToolMessage is "
        "emitted; absent when ``interrupt()`` pauses the graph before "
        "the tool completes). Use ``X.get('output')`` instead — the "
        "downstream ``insert_tool(..., output: Optional[str])`` "
        "accepts ``None`` and SQLite stores it as NULL."
    )


# ============================================================================
# Sentinel 2: chat_bot_node uses ``.get('output')`` at least once
#             (the positive structural assertion that the fix is in place).
# ============================================================================


def test_chat_bot_node_uses_get_for_output_access():
    """At least one ``<Name>.get('output', ...)`` call must exist in
    ``chat_bot_node`` — the principled accessor that tolerates the
    paused-interrupt case where a tool_call has no ``output`` key."""
    func = _chat_bot_node_funcdef()
    assert func is not None

    found = False
    for node in ast.walk(func):
        if _is_output_get_call(node):
            found = True
            break

    assert found, (
        "chat_bot_node must use ``<Name>.get('output')`` (or "
        "``.get('output', default)``) somewhere — the typed Optional "
        "accessor matching ``sqlite_handler.insert_tool``'s "
        "``output: Optional[str]`` signature."
    )


# ============================================================================
# Sentinel 3: insert_tool downstream contract is Optional[str] for output
#             (locks the upstream-side justification that None is valid).
# ============================================================================


def test_insert_tool_signature_accepts_optional_output():
    """The fix's principle rests on the typed contract of
    ``SQLiteHandler.insert_tool``: ``output`` is ``Optional[str]``.
    Lock the signature so a future refactor cannot tighten it to
    ``str`` (which would re-introduce the Bug-O failure mode at the
    boundary)."""
    from simulator.utils.sqlite_handler import SqliteSaver

    sig = inspect.signature(SqliteSaver.insert_tool)
    output_param = sig.parameters.get("output")
    assert output_param is not None, (
        "SQLiteHandler.insert_tool must accept an ``output`` "
        "parameter — refactor changed the signature."
    )

    # The annotation may be ``Optional[str]``, ``str | None``, or the
    # ``Union[str, None]`` form. Check via typing.get_args / Union
    # introspection rather than string-matching the annotation.
    import typing
    annot = output_param.annotation
    if annot is inspect.Parameter.empty:
        # No annotation — strictly speaking, accepts anything; treat
        # as compatible but flag for future tightening.
        return
    # For ``Optional[str]`` / ``str | None`` / ``Union[str, None]``:
    # typing.get_args returns the constituent types.
    args = typing.get_args(annot)
    accepts_none = (
        type(None) in args
        or annot is type(None)
    )
    assert accepts_none, (
        f"SQLiteHandler.insert_tool.output annotation is {annot!r} — "
        "must accept ``None`` (typically ``Optional[str]``) so paused "
        "tool_calls can be persisted with NULL output. Tightening "
        "this annotation would re-introduce Bug O at the boundary."
    )
