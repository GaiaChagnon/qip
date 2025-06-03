# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Cartpole with four pendulum links (quadruple)."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Update to your actual quadruple cartpole USD path:
        usd_path="C:/Users/Shadow/cartpole_quadruple.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 4.2),
        joint_pos={
            "slider_to_cart": 0.0,
            "cart_to_pole": 0.0,
            "pole_to_pole": 0.0,
            "pole_to_pole_to_pole": 0.0,
            "pole_to_pole_to_pole_to_pole": 0.0,
        }
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit=600.0,
            velocity_limit=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit=100.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=0.0018,
            friction=0.0,
        ),
        "pole_to_pole": ImplicitActuatorCfg(
            joint_names_expr=["pole_to_pole"],
            effort_limit=100.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=0.00012,
            friction=0.0,
        ),
        "pole_to_pole_to_pole": ImplicitActuatorCfg(
            joint_names_expr=["pole_to_pole_to_pole"],
            effort_limit=100.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=0.0009,
            friction=0.0,
        ),
        "pole_to_pole_to_pole_to_pole": ImplicitActuatorCfg(
            joint_names_expr=["pole_to_pole_to_pole_to_pole"],
            effort_limit=100.0,
            velocity_limit=500.0,
            stiffness=0.0,
            damping=0.0006,
            friction=0.0,
        ),
    },
)
"""Configuration for a Cartpole robot with four pendulum links."""