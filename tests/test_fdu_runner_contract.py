"""Static contracts that keep the WYW/Fudan sequence runner lifecycle aligned."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def test_sequence_runner_randomizes_episode_lengths_after_final_reset():
    path = ROOT / "source/agent_rl/agent_rl/rsl_rl/runners/on_policy_sequence_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnPolicySequenceRunner"
    )
    learn = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "learn"
    )

    reset_line = min(
        node.lineno
        for node in ast.walk(learn)
        if isinstance(node, ast.Call)
        and _attribute_path(node.func) == ("self", "env", "reset")
    )
    randomize_line = min(
        node.lineno
        for node in ast.walk(learn)
        if isinstance(node, ast.Assign)
        and any(
            _attribute_path(target) == ("self", "env", "episode_length_buf")
            for target in node.targets
        )
    )
    assert reset_line < randomize_line


def test_training_enables_initial_episode_length_randomization():
    path = ROOT / "scripts/rsl_rl/train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_path(node.func) == ("runner", "learn")
    ]
    assert len(calls) == 1
    keyword = next(
        item for item in calls[0].keywords if item.arg == "init_at_random_ep_len"
    )
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is True
