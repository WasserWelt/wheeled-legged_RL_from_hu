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


def test_sequence_runner_owns_compact_tensorboard_routing_and_console_filter():
    path = ROOT / "source/agent_rl/agent_rl/rsl_rl/runners/on_policy_sequence_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnPolicySequenceRunner"
    )
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "log" for node in runner.body
    )
    source = path.read_text(encoding="utf-8")
    assert '("Episode/Reward/", "RewardTerms/")' in source
    assert '("Episode/Reset/", "Termination/Count/")' in source
    assert '("Episode/FDU_L0Boundary/", "FDU/L0/Episode/")' in source
    assert "Diagnostics/FDU_L0Boundary/" not in source
    assert "_SEQUENCE_CONSOLE_KEYS" in source
    assert 'key.startswith("Episode/Reward/")' in source
    assert 'f"{\'ETA:\':>{pad}}' in source


def test_sequence_encoder_receives_done_mask_and_emits_distribution_diagnostics():
    algorithm_path = ROOT / "source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_sequence.py"
    storage_path = ROOT / "source/agent_rl/agent_rl/rsl_rl/storage/rollout_storage_sequence.py"
    algorithm_source = algorithm_path.read_text(encoding="utf-8")
    storage_source = storage_path.read_text(encoding="utf-8")
    assert "encoder_exclude_terminal" in algorithm_source
    assert "F.smooth_l1_loss" in algorithm_source
    assert 'stats("target", target)' in algorithm_source
    assert 'stats("latent", prediction)' in algorithm_source
    assert 'stats("error", error)' in algorithm_source
    assert 'f"encoder_{prefix}_rms"' in algorithm_source
    assert 'f"encoder_{prefix}_p99_abs"' in algorithm_source
    assert 'f"encoder_{prefix}_max_abs"' in algorithm_source
    assert 'diagnostics["encoder_terminal_mse"]' in algorithm_source
    assert "dones_batch" in storage_source


def test_wyw_runner_starts_with_bounded_actions_and_robust_encoder():
    path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/agents/rsl_rl_ppo_cfg.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WheelbipeWywPPORunnerCfg"
    )
    clip = next(
        node
        for node in runner.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "clip_actions" for target in node.targets)
    )
    assert ast.literal_eval(clip.value) == 1.0
    assert '"extra_learning_rate": 1.0e-4' in source
    assert '"encoder_loss": "smooth_l1"' in source
    assert '"encoder_exclude_terminal": True' in source
