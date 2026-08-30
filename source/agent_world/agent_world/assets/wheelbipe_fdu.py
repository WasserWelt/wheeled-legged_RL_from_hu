# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================

"""ArticulationCfg for the Fudan closed-loop (五连杆 / 筝形) wheeled-legged robot.

The source of truth is ``meshes/infantry_V2.urdf``. The policy drives the front
and rear entity hip joints directly; it does not use Fudan's old virtual-knee
torque mapping. The parallel loop is closed in the USD by 4 spherical loop joints flagged
``physics:excludeFromArticulation`` (see robot_models/.../add_loop_joints.py).

Joint roles (14 DOF total):

  driven hips (legs_act)   : rf0_Joint, lf0_Joint  (front bar, = hu front1)
                             r20_Joint, l20_Joint  (rear bar,  = hu rear1)
  wheels (wheel)           : r_wheel_Joint, l_wheel_Joint
  passive linkage (inact)  : rf1_Joint, lf1_Joint          (front knee)
                             r21/r22/r23_Joint, l21/l22/l23_Joint (rear linkage,
                                                           constrained by the loops)
The selected URDF's base mesh and inertial block are used verbatim.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

from agent_world import AssetPath


# Exact vertical equivalent-leg solution for infantry_V2.urdf:
# pi/2 - atan2(coupler=0.208, crank=0.17472).
_VERTICAL_JOINT_CENTER = math.pi / 2.0 - math.atan2(0.208, 0.17472)


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
        # PhysX scan: this symmetric pose gives theta0=0 and puts the 0.06 m
        # wheels on a flat floor at root z ~=0.245--0.249 m.  Passive joints
        # remain at the imported loop assembly zeros and settle within the
        # loop tolerance; only the four physical drive bars are commanded.
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "rf0_Joint": -_VERTICAL_JOINT_CENTER,
            "lf0_Joint": _VERTICAL_JOINT_CENTER,
            "r20_Joint": -_VERTICAL_JOINT_CENTER,
            "l20_Joint": _VERTICAL_JOINT_CENTER,
            "rf1_Joint": 0.0,
            "lf1_Joint": 0.0,
            "r2[123]_Joint": 0.0,
            "l2[123]_Joint": 0.0,
            ".*_wheel_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Four entity motors: front and rear driven bar on each closed-chain leg.
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=["lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint"],
            stiffness=20.0,
            damping=1.0,
            effort_limit=40.0,
            velocity_limit=30.0,
            armature=0.0,
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
            # Fudan Plane training clamps wheel torque at 5 N m. Jump's
            # training URDF uses 50 N m and overrides this in its env cfg.
            # (The 5/4 N m sim2sim/deployment clamps are a separate stage.)
            effort_limit=5.0,
            # The Fudan controller has no wheel-speed clamp: it applies a
            # torque-limited velocity error. 60 rad/s covers its command range
            # on the 0.06 m wheels while keeping this target-control adapter
            # bounded. Keep the PhysX limit identical to the runtime clamp.
            velocity_limit=60.0,
            velocity_limit_sim=60.0,
            armature=0.0,
        ),
    },
)
