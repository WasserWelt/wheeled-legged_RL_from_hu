"""Static and numerical contracts for WYW/Fudan control and done semantics."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch


ROOT = Path(__file__).parents[2]


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


def _event_params(path: Path, class_name: str, event_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == event_name for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Call)
    params = next(keyword.value for keyword in assignment.value.keywords if keyword.arg == "params")
    assert isinstance(params, ast.Dict)
    return {
        ast.literal_eval(key): ast.literal_eval(value)
        for key, value in zip(params.keys, params.values)
        if isinstance(key, ast.Constant) and key.value != "asset_cfg"
    }


def test_wyw_wheel_target_uses_full_normalized_range_and_p13_71_speed_cap():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    target_limit = _class_assignment(env_cfg_path, "WheelbipeWywFlatEnvCfg", "max_wheel_vel")
    target_scale = _class_assignment(
        env_cfg_path, "WheelbipeWywFlatEnvCfg", "wheel_vel_action_scale"
    )
    assert target_limit == 60.0
    assert target_scale == 60.0

    # A normalized action spans the full protected wheel-target range, and
    # larger policy outputs remain capped at the same physical velocity.
    actions = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    wyw_target = torch.clamp(target_scale * actions, -target_limit, target_limit)
    assert torch.equal(wyw_target, torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0]))

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


def test_wyw_default_height_command_matches_documented_value():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    default_height = _class_assignment(
        env_cfg_path, "WheelbipeWywFlatEnvCfg", "default_height_cmd"
    )
    assert default_height == 0.22


def test_wyw_flat_material_contract_separates_wheels_links_and_ground():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    params = _event_params(env_cfg_path, "FduEventCfg", "physics_material")
    assert params == {
        "friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.6, 0.9),
        "restitution_range": (0.10, 0.45),
        "link_static_friction_range": (0.6, 1.4),
        "link_restitution_range": (0.05, 0.20),
    }

    source = env_cfg_path.read_text(encoding="utf-8")
    assert "cfg.terrain.physics_material.static_friction = 0.65" in source
    assert "cfg.terrain.physics_material.dynamic_friction = 0.55" in source
    assert "cfg.terrain.physics_material.restitution = 0.175" in source

    tree = ast.parse(source)
    randomizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "randomize_fdu_material"
    )
    randomizer_source = ast.unparse(randomizer)
    assert "wheel_shape_indices" in randomizer_source
    assert "link_shape_indices" in randomizer_source
    assert "wheel_dynamic = torch.minimum(wheel_dynamic, wheel_static)" in randomizer_source


def test_fdu_asset_uses_500hz_wrapped_difference_for_all_actuators():
    asset_path = ROOT / "source/agent_world/agent_world/assets/wheelbipe_fdu.py"
    tree = ast.parse(asset_path.read_text(encoding="utf-8"))
    actuators = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "DiffVelPDActuatorCfg"
    ]
    assert len(actuators) == 3
    for actuator in actuators:
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in actuator.keywords}
        assert values["diff_dt"] == 0.002
        assert values["wrap_to_pi"] is True

    wheel_actuator = next(
        actuator for actuator in actuators
        if any(
            keyword.arg == "joint_names_expr"
            and ast.literal_eval(keyword.value) == [".*_wheel_Joint"]
            for keyword in actuator.keywords
        )
    )
    values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in wheel_actuator.keywords}
    assert values["damping"] == 0.2
    assert values["effort_limit"] == 5.0
    assert values["velocity_limit"] == 60.0
    assert values["velocity_limit_sim"] == 60.0


def test_wyw_velocity_contract_reaches_observation_reward_and_critic():
    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    source = env_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WheelbipeWywEnv"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in env_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "policy_vel = self._wyw_joint_velocity" in methods["_build_wyw_policy_obs"]
    assert "joint_acc = self._wyw_policy_joint_acceleration" in methods["_build_wyw_critic_obs"]
    reward_source = methods["_compute_fdu_reward_terms"]
    assert "qdot = self._update_wyw_joint_velocity(final_sample_id)" in reward_source
    assert "qddot = self._update_wyw_policy_joint_acceleration(final_sample_id)" in reward_source
    assert "self.robot.data.joint_vel[:, self._actuate_idx]" not in reward_source
    assert "self.robot.data.joint_acc[:, self._actuate_idx]" not in reward_source
    acceleration_source = methods["_update_wyw_policy_joint_acceleration"]
    assert "self._wyw_last_policy_joint_velocity - velocity" in acceleration_source

    cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    assert _class_assignment(
        cfg_path, "WheelbipeWywFlatEnvCfg", "wyw_joint_velocity_source"
    ) == "wrapped_position_difference"
    assert _class_assignment(
        cfg_path, "WheelbipeWywFlatEnvCfg", "wyw_joint_velocity_diff_dt"
    ) == 0.002


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


def test_flat_keeps_inherited_plane_and_rough_keeps_generator():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    tree = ast.parse(env_cfg_path.read_text(encoding="utf-8"))

    common_helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_wyw_common"
    )
    common_source = ast.unparse(common_helper)
    assert "terrain_type" not in common_source
    assert "flat_ground.usda" not in common_source

    rough_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WheelbipeWywRoughEnvCfg"
    )
    rough_post_init = next(
        node
        for node in rough_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    terrain_type_assignments = [
        node
        for node in rough_post_init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "terrain_type"
            for target in node.targets
        )
    ]
    assert len(terrain_type_assignments) == 1
    assert ast.literal_eval(terrain_type_assignments[0].value) == "generator"


def test_wyw_done_reasons_are_latched_and_logged_separately():
    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    source = env_path.read_text(encoding="utf-8")
    for reason in (
        "orientation",
        "contact",
        "persistent_failure",
        "numerical_safety",
        "terrain_boundary",
    ):
        assert f"_wyw_done_reason_{reason}" in source
        assert f'"Termination/Count/{{name}}"' in source
        assert f'"Termination/Fraction/{{name}}"' in source
    assert "Termination/ContactBody/" in source


def test_persistent_failure_has_unclipped_terminal_penalty():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    tree = ast.parse(env_cfg_path.read_text(encoding="utf-8"))
    expected_termination = {
        "FDU_PLANE_REWARDS": -500.0,
        "FDU_JUMP_REWARDS": -200.0,
    }
    for rewards_name in ("FDU_PLANE_REWARDS", "FDU_JUMP_REWARDS"):
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == rewards_name for target in node.targets)
        )
        values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in assignment.value.keywords
        }
        assert values["termination"] == expected_termination[rewards_name]

    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    env_source = env_path.read_text(encoding="utf-8")
    assert 'self.cfg.rewards.get("termination", 0.0)' in env_source
    assert '"_wyw_failure_termination_reward_mask"' in env_source
    for required_mask in (
        "persistent_failure",
        "terminate",
        "~time_out",
        "~immediate_terminate",
        "~boundary_terminate",
    ):
        assert required_mask in env_source


def test_wheel_contact_loss_is_flat_only_and_uses_two_frame_filter():
    cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    cfg_source = cfg_path.read_text(encoding="utf-8")
    assert 'FDU_FLAT_REWARDS["wheel_contact_loss"] = -1.0' in cfg_source
    assert "wyw_wheel_contact_reward_enabled = True" in cfg_source
    assert cfg_source.count("wyw_wheel_contact_reward_enabled = False") == 2

    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    env_source = env_path.read_text(encoding="utf-8")
    assert "contact_now | self._wyw_last_reward_wheel_contacts" in env_source
    assert 'terms["wheel_contact_loss"] = compute_fdu_wheel_contact_loss' in env_source


def test_flat_command_curriculum_starts_at_slow_speed():
    env_cfg_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py"
    source = env_cfg_path.read_text(encoding="utf-8")
    # Flat and Rough each explicitly restore the slow initial range after the
    # shared helper installs the task-wide command plumbing.
    assert source.count("self.commands.ranges.lin_vel_x = (-0.5, 0.5)") == 2


def test_l0_tensorboard_logging_keeps_only_episode_affected_fraction():
    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    source = env_path.read_text(encoding="utf-8")
    assert '"Episode/FDU_L0Boundary/affected_env_fraction"' in source
    for removed_key in (
        "Episode/FDU_L0Boundary/mean_physics_samples",
        "Episode/FDU_L0Boundary/entry_events",
        "Episode/FDU_L0Boundary/min_measured_l0_m",
        "Diagnostics/FDU_L0Boundary/current_env_count",
        "Diagnostics/FDU_L0Boundary/current_env_fraction",
        "Diagnostics/FDU_L0Boundary/current_min_measured_l0_m",
        "Diagnostics/FDU_L0Boundary/global_min_measured_l0_m",
        "Diagnostics/FDU_L0Boundary/total_physics_samples",
        "Diagnostics/FDU_L0Boundary/total_entry_events",
    ):
        assert removed_key not in source
