"""Isaac Lab asset configuration for the old_chuan_V2 closed-chain robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath


Wheelbipe_OldChuanV2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/wheelbipe_old_chuan_v2/old_chuan_V2.usd",
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        # Start from the imported assembly pose.  Validation scripts pre-shape
        # the four motors after reset; this avoids overlapping regex/exact-name
        # entries in Isaac Lab's initial-state resolver.
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[
                "left_front_1_joint", "left_rear_1_joint",
                "right_front_1_joint", "right_rear_1_joint",
            ],
            stiffness=20.0,
            damping=1.0,
            effort_limit=40.0,
            velocity_limit=30.0,
            armature=0.0,
        ),
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_front_2_joint", ".*_front_3_joint", ".*_front_4_joint",
                ".*_rear_2_joint",
            ],
            stiffness=0.0,
            damping=0.01,
            effort_limit=100.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=0.002,
            effort_limit=40.0,
            velocity_limit=300.0,
            armature=0.0,
        ),
    },
)
