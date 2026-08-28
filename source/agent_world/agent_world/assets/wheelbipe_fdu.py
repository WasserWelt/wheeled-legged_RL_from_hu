# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================

"""ArticulationCfg for the Fudan closed-loop (五连杆 / 筝形) wheeled-legged robot.

Migrated from the Fudan MuJoCo model. Unlike the serial-virtual-leg + mapped-torque
scheme Fudan used at deploy time, here we train NATIVELY on the closed chain: each
leg drives two hip joints (front + rear bar), exactly like hu's parallel-linkage
model. The parallel loop is closed in the USD by 4 spherical loop joints flagged
``physics:excludeFromArticulation`` (see robot_models/.../add_loop_joints.py).

Joint roles (16 DOF total), and how they map onto hu's Wheelbipe_V14_2 groups:

  driven hips (legs_act)   : rf0_Joint, lf0_Joint  (front bar, = hu front1)
                             r20_Joint, l20_Joint  (rear bar,  = hu rear1)
  wheels (wheel)           : r_wheel_Joint, l_wheel_Joint
  passive linkage (inact)  : rf1_Joint, lf1_Joint          (front knee)
                             r21/r22/r23_Joint, l21/l22/l23_Joint (rear linkage,
                                                           constrained by the loops)
  dummy gas-spring (spring): left_spring2_joint, right_spring2_joint
                             -- decoupled prismatic DOFs on the base, kept so hu's
                                spring pipeline has a slot; given ZERO drive and
                                ZERO damping so they never affect the leg.

The gimbal (云台 yaw/pitch) is intentionally absent here -- base_link_del is the
gimbal-removed base; the gimbal is added as a separate later step.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

from agent_world import AssetPath

# Fudan hips use DM-series actuators; reuse hu's DM8009 reflected armature as a
# reasonable default (gear ratio 9:1). Tune once real motor specs are confirmed.
DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0


Wheelbipe_FDU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/wheelbipe_fdu/wheelbipe_fdu.usd",
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
            # closed loop -> lean on more position iterations for constraint stability
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # zero-joint pose is the assembly config the loops were anchored at.
        # spawn height is a placeholder -- tune so the wheels sit near the ground.
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "rf0_Joint": 0.0,
            "lf0_Joint": 0.0,
            "r20_Joint": 0.0,
            "l20_Joint": 0.0,
            "rf1_Joint": 0.0,
            "lf1_Joint": 0.0,
            "r2[123]_Joint": 0.0,
            "l2[123]_Joint": 0.0,
            ".*_wheel_Joint": 0.0,
            ".*_spring2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # driven hips: front bar (rf0/lf0) + rear bar (r20/l20)
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=["rf0_Joint", "lf0_Joint", "r20_Joint", "l20_Joint"],
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=DM8009_ARMATURE,
        ),
        # passive linkage joints (front knee + rear 3-bar); the loop constraints
        # determine their motion, so no drive -- just a whisker of damping.
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=["rf1_Joint", "lf1_Joint", "r2[123]_Joint", "l2[123]_Joint"],
            stiffness=0.0,
            damping=0.01,
            effort_limit=50.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_Joint"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.0,
        ),
        # dummy gas-spring DOFs: no force, no damping -> zero effect on the leg.
        "spring": IdealPDActuatorCfg(
            joint_names_expr=[".*_spring2_joint"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=0.0,
            velocity_limit=50.0,
            armature=0.0001,
        ),
    },
)
