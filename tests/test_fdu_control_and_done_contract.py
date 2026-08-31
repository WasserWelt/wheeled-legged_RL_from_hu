"""Static and numerical contracts for WYW/Fudan control and done semantics."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _class_assignment(path: Path, class_name: str, attribute: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == attribute for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_wyw_wheel_pd_path_and_p13_71_speed_cap():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    target_limit = _class_assignment(env_cfg_path, "WheelbipeWywFlatEnvCfg", "max_wheel_vel")
    target_scale = _class_assignment(
        env_cfg_path, "WheelbipeWywFlatEnvCfg", "wheel_vel_action_scale"
    )
    assert target_limit == 60.0
    assert target_scale == 10.0

    # In the unsaturated operating range, the velocity-target adapter computes
    # exactly the same torque as Fudan's explicit wheel PD expression.
    actions = torch.tensor([-6.0, -2.0, 0.0, 3.0, 6.0])
    wheel_vel = torch.tensor([-70.0, -30.0, 1.0, 45.0, 70.0])
    fudan_torque = torch.clamp(0.2 * (10.0 * actions - wheel_vel), -5.0, 5.0)
    wyw_target = torch.clamp(target_scale * actions, -target_limit, target_limit)
    wyw_ideal_pd_torque = torch.clamp(0.2 * (wyw_target - wheel_vel), -5.0, 5.0)
    assert torch.equal(wyw_ideal_pd_torque, fudan_torque)

    # Convert the two P19 curve anchors supplied for the C620 to P13.71. The
    # selected 60 rad/s hard cap stays below both converted operating speeds.
    speed_scale = 19.0 / 13.71
    torque_scale = 13.71 / 19.0
    p13_no_load_rad_s = 500.0 * speed_scale * 2.0 * torch.pi / 60.0
    p13_loaded_rad_s = 450.0 * speed_scale * 2.0 * torch.pi / 60.0
    p13_loaded_torque = 4.5 * torque_scale
    assert p13_no_load_rad_s > target_limit
    assert p13_loaded_rad_s > target_limit
    assert math.isclose(p13_loaded_torque, 3.2471052631578947)


def test_wheel_asset_keeps_p13_71_conservative_velocity_limit():
    asset_path = ROOT / "source/agent_world/agent_world/assets/wheelbipe_fdu.py"
    tree = ast.parse(asset_path.read_text(encoding="utf-8"))
    wheel_actuator = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "IdealPDActuatorCfg"
        and any(
            keyword.arg == "joint_names_expr"
            and ast.literal_eval(keyword.value) == [".*_wheel_Joint"]
            for keyword in node.keywords
        )
    )
    values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in wheel_actuator.keywords}
    assert values["damping"] == 0.2
    assert values["effort_limit"] == 5.0
    assert values["velocity_limit"] == 60.0
    assert values["velocity_limit_sim"] == 60.0


def test_rough_boundary_is_immediate_termination_not_timeout():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    boundary_cfg = _class_assignment(
        env_cfg_path,
        "WheelbipeWywRoughEnvCfg",
        "rough_terrain_boundary_reset_cfg",
    )
    assert boundary_cfg == {
        "enabled": True,
        "margin": 1.0,
        "use_inner_terrain_area": False,
    }

    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    tree = ast.parse(env_path.read_text(encoding="utf-8"))
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WheelbipeWywEnv"
    )
    get_dones = next(
        node
        for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_dones"
    )
    boundary_assignment = next(
        node
        for node in get_dones.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "boundary_terminate" for target in node.targets)
    )
    assert any(
        isinstance(node, ast.Call) and _call_name(node) == "_get_rough_terrain_boundary_termination"
        for node in ast.walk(boundary_assignment.value)
    )

    terminate_assignment = next(
        node
        for node in get_dones.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "terminate" for target in node.targets)
    )
    assert "boundary_terminate" in {
        node.id for node in ast.walk(terminate_assignment.value) if isinstance(node, ast.Name)
    }
    assert not any(
        isinstance(node, (ast.Assign, ast.AugAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "time_out"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        and any(
            isinstance(child, ast.Name) and child.id == "boundary_terminate"
            for child in ast.walk(node)
        )
        for node in ast.walk(get_dones)
    )
