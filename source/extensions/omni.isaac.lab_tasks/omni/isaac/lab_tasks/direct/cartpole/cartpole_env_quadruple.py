# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# Import the articulation config for quadruple pole
from omni.isaac.lab_assets.cartpole_quadruple import CARTPOLE_CFG  # or wherever your .usd is located

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

#Domain randomization : 
from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg
from omni.isaac.lab.envs.mdp import events as mdp
from omni.isaac.lab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg
from omni.isaac.lab.utils.noise import gaussian_noise

import logging
import sys
import gymnasium as gym

logger = logging.getLogger("cartpole_env_quadruple")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    file_handler = logging.FileHandler("cartpole_env_quadruple.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Write to real stdout
    stdout_handler = logging.StreamHandler(sys.__stdout__)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    """
    Example config for a quadruple-pole cart environment.
    """

    debug = False
    debug_env_index = 1

    decimation = 2
    episode_length_s = 15.0
    action_scale = 40.0
    action_space = 1
    observation_space = 10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 300,
        render_interval=decimation
    )

    # Robot / Articulation
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"                          # 1st pole
    pole01_dof_name = "pole_to_pole"                        # 2nd pole
    pole02_dof_name = "pole_to_pole_to_pole"                # 3rd pole
    pole03_dof_name = "pole_to_pole_to_pole_to_pole"        # 4th pole

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=5.0,
        replicate_physics=True
    )

    max_cart_pos = 0.65

    # Initial angles for each of the four poles
    initial_pole_angle_range = [0.0, 1.0] # 1st pole
    initial_pole01_angle_range = [-0.04, 0.04]    # 2nd pole
    initial_pole02_angle_range = [-0.03, 0.03]    # 3rd pole
    initial_pole03_angle_range = [-0.02, 0.02]  # 4th pole

    # Small random initial velocity for all poles
    initial_velocity_range = [-0.05, 0.05]

    rew_scale_terminated = -6000.0
    bonus_cycle_length: int = 40

    # Additional reward scales (tune to your liking)
    rew_scale_alive = 0
    rew_scale_pole_angle = 0
    rew_scale_cart_vel = 0
    rew_scale_pole_vel = 0
    rew_scale_cart_center = 0
    rew_scale_pole_straight = 0
    rew_scale_pole_upwards = 0
    rew_scale_exp_vel = 0
    vel_scale_factor = 0
    rew_scale_bonus_first = 0
    rew_scale_bonus_second = 0
    rew_scale_bonus_third = 0

    # New parameters for the "too low" termination condition
    low_height_threshold = -5.0  # Height threshold below which we count time
    low_height_timeout_s = 5.5  # Time in seconds to allow being below threshold

    # Domain randomization enable/disable flags
    randomization_enable_joint_damping = False
    randomization_enable_mass_properties = False
    randomization_enable_gravity = False
    randomization_enable_observation_noise = False # NEW: Flag for observation noise
    randomization_enable_action_noise = False       # NEW: Flag for action noise

    observation_noise_std = 0.001  # Standard deviation for observation noise
    action_noise_std = 0.03        # Standard deviation for action noise

    
    # Domain randomization events
    events = {
        # Randomize damping of the pendulums
        "pendulum_damping": EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=["cart_to_pole"]),
                "damping_distribution_params": (0.0016, 0.0020),  # Around 0.0018
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        
        "pendulum_damping_pole1": EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=["pole_to_pole"]),
                "damping_distribution_params": (0.0001, 0.00014),  # Around 0.00012
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        
        "pendulum_damping_pole2": EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=["pole_to_pole_to_pole"]),
                "damping_distribution_params": (0.0008, 0.0010),  # Around 0.0009
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        
        "pendulum_damping_pole3": EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", joint_names=["pole_to_pole_to_pole_to_pole"]),
                "damping_distribution_params": (0.0005, 0.0007),  # Around 0.0006
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        
        # Randomize mass properties with Gaussian distribution for each specific pole
        "pole1_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", body_names=["pole"]),
                "mass_distribution_params": (0.21, 0.02),  # Mean 0.3 kg, std 0.01 kg
                "operation": "abs",
                "distribution": "gaussian",
            },
        ),
        
        "pole2_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", body_names=["pole01"]),
                "mass_distribution_params": (0.115, 0.02),  # Mean 0.25 kg, std 0.01 kg
                "operation": "abs",
                "distribution": "gaussian",
            },
        ),
        
        "pole3_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", body_names=["pole02"]),
                "mass_distribution_params": (0.147, 0.02),  # Mean 0.2 kg, std 0.01 kg
                "operation": "abs",
                "distribution": "gaussian",
            },
        ),
        
        "pole4_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cartpole", body_names=["pole03"]),
                "mass_distribution_params": (0.125, 0.02),  # Mean 0.15 kg, std 0.01 kg
                "operation": "abs",
                "distribution": "gaussian",
            },
        ),
        
        # Randomize gravity more frequently (every ~10 seconds)
        "gravity_randomization": EventTermCfg(
            func=mdp.randomize_physics_scene_gravity,
            mode="interval",
            is_global_time=True,
            interval_range_s=(8.0, 12.0),  # Randomize gravity every ~10 seconds
            params={
                "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, 0.5]),
                "operation": "add",
                "distribution": "gaussian",
            },
        ),
    }



class CartpoleQuadrupleEnv(DirectRLEnv):
    """
    Quadruple-pole cart environment, adapted from the triple-pole version.
    We now have four pendulum DOFs and incorporate them in the observation,
    reward, and reset logic.
    """

    cfg: CartpoleQuadrupleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
           
        # Apply domain randomization enable/disable flags
        if hasattr(self.cfg, 'events') and hasattr(self.cfg, 'randomization_enable_joint_damping') and not self.cfg.randomization_enable_joint_damping:
            for attr in ["pendulum_damping", "pendulum_damping_pole1", 
                        "pendulum_damping_pole2", "pendulum_damping_pole3"]:
                if attr in self.cfg.events:
                    del self.cfg.events[attr]
        
        if hasattr(self.cfg, 'events') and hasattr(self.cfg, 'randomization_enable_mass_properties') and not self.cfg.randomization_enable_mass_properties:
            for attr in ["pole1_mass", "pole2_mass", "pole3_mass", "pole4_mass"]:
                if attr in self.cfg.events:
                    del self.cfg.events[attr]
        
        if hasattr(self.cfg, 'events') and hasattr(self.cfg, 'randomization_enable_gravity') and not self.cfg.randomization_enable_gravity:
            if "gravity_randomization" in self.cfg.events:
                del self.cfg.events["gravity_randomization"]

        # DOF indices by name
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self._pole01_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole01_dof_name)
        self._pole02_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole02_dof_name)
        self._pole03_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole03_dof_name)

        self.episode_info = {}
        self.action_scale = self.cfg.action_scale

        # For counting how many steps we've been "balanced"
        self.balance_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.previously_balanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # For tracking how long each env has been below height threshold
        self.low_height_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Convert timeout from seconds to steps
        self.low_height_timeout_steps = int(self.cfg.low_height_timeout_s / (self.cfg.sim.dt * self.cfg.decimation))
        
        # Initialize storage for previous pole states (for delayed observations)
        self._prev_pole_pos = None
        self._prev_pole_vel = None
        self._prev_pole01_pos = None
        self._prev_pole01_vel = None
        self._prev_pole02_pos = None
        self._prev_pole02_vel = None
        self._prev_pole03_pos = None
        self._prev_pole03_vel = None

        # For four links, if each is length 1.0, the tip is near 4.0 in height at perfect upright
        self.balance_threshold = {
            'height': 3.40,  # slightly below 4.0 if "perfectly upright" is around 4.0
        }
        self.min_steps_balanced = 120
        self.total_successful_envs = 0


    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Performs one pre-physics step:
        1) Converts the agent's raw action into the scaled acceleration for the cart DOF.
        2) Tracks how long the environment remains 'balanced' based on the poles' tip height.
        3) Tracks how long the environment has been below the minimum height threshold.
        4) Optionally logs debug info for a single environment (acceleration, velocity, etc.).
        """

        # 1) Convert the agent’s output into the scaled acceleration
        actions = actions.squeeze(-1)  # shape [num_envs]

        
        # Apply action noise if enabled
        if hasattr(self.cfg, 'randomization_enable_action_noise') and self.cfg.randomization_enable_action_noise:
            # Create a noise config for the action noise
            noise_cfg = GaussianNoiseCfg(mean=0.0, std=self.cfg.action_noise_std, operation="scale")
            # Apply noise directly using the gaussian_noise function
            actions = gaussian_noise(actions, noise_cfg)

        acceleration = torch.clamp(actions, -1.0, 1.0) * self.cfg.action_scale
        acceleration_2d = acceleration.unsqueeze(-1)  # shape [num_envs, 1]

        # 2) Read all joint positions for tip-height check
        pole1_angle = self.cartpole.data.joint_pos[:, self._pole_dof_idx[0]]
        pole2_angle = self.cartpole.data.joint_pos[:, self._pole01_dof_idx[0]]
        pole3_angle = self.cartpole.data.joint_pos[:, self._pole02_dof_idx[0]]
        pole4_angle = self.cartpole.data.joint_pos[:, self._pole03_dof_idx[0]]

        L0 = L1 = L2 = L3 = 1.0
        tip_height = (
            L0 * torch.cos(pole1_angle)
            + L1 * torch.cos(pole1_angle + pole2_angle)
            + L2 * torch.cos(pole1_angle + pole2_angle + pole3_angle)
            + L3 * torch.cos(pole1_angle + pole2_angle + pole3_angle + pole4_angle)
        )

        # Check which environments are "balanced" based on tip height
        balanced = tip_height > self.balance_threshold['height']

        # Update consecutive balance counter
        self.balance_counter[balanced] += 1
        self.balance_counter[~balanced] = 0

        # Check which envs are currently successful
        currently_successful = self.balance_counter >= self.min_steps_balanced

        # Count new successes (environments that just became successful)
        new_successes = currently_successful & ~self.previously_balanced
        self.total_successful_envs += new_successes.sum().item()

        # Update memory for next step
        self.previously_balanced = currently_successful
        self.episode_info["final_balance_count"] = self.total_successful_envs

        # 2.1) Update low height counter
        too_low = tip_height < self.cfg.low_height_threshold
        self.low_height_counter[too_low] += 1
        self.low_height_counter[~too_low] = 0

        # 3) (Optional) Debug logging for one specific environment
        # 3) (Optional) Debug logging for one specific environment
        if self.cfg.debug:
            debug_env = self.cfg.debug_env_index
            if debug_env < self.num_envs:
                # Cart data (already implemented)
                cart_pos_debug = self.cartpole.data.joint_pos[debug_env, self._cart_dof_idx[0]].item()
                cart_vel_debug = self.cartpole.data.joint_vel[debug_env, self._cart_dof_idx[0]].item()
                acc_debug = acceleration[debug_env].item()
                
                # Extract pole angles for debug environment
                p1_angle_debug = pole1_angle[debug_env].item()
                p2_angle_debug = pole2_angle[debug_env].item()
                p3_angle_debug = pole3_angle[debug_env].item()
                p4_angle_debug = pole4_angle[debug_env].item()
                
                # Extract pole velocities for debug environment
                p1_vel_debug = self.cartpole.data.joint_vel[debug_env, self._pole_dof_idx[0]].item()
                p2_vel_debug = self.cartpole.data.joint_vel[debug_env, self._pole01_dof_idx[0]].item()
                p3_vel_debug = self.cartpole.data.joint_vel[debug_env, self._pole02_dof_idx[0]].item()
                p4_vel_debug = self.cartpole.data.joint_vel[debug_env, self._pole03_dof_idx[0]].item()
                
                # Log all data with clear labels
                logger.debug(
                    f"[Env={debug_env}] "
                    f"Acc={acc_debug:.3f}, "
                    f"Cart: pos={cart_pos_debug:.3f}, vel={cart_vel_debug:.3f} | "
                    f"P1: angle={p1_angle_debug:.3f}, vel={p1_vel_debug:.3f} | "
                    f"P2: angle={p2_angle_debug:.3f}, vel={p2_vel_debug:.3f} | "
                    f"P3: angle={p3_angle_debug:.3f}, vel={p3_vel_debug:.3f} | "
                    f"P4: angle={p4_angle_debug:.3f}, vel={p4_vel_debug:.3f}"
                )

        # 4) Apply the computed acceleration to the cart DOF
        self.cartpole.set_joint_effort_target(acceleration_2d, joint_ids=self._cart_dof_idx)

        # 5) Write all updated data to the simulator
        self.cartpole.write_data_to_sim()

    def _apply_action(self) -> None:
        # Overridden from parent – no additional logic needed here,
        # everything was done in _pre_physics_step
        self.cartpole.write_data_to_sim()
        pass

    def _get_observations(self) -> dict:
        """
        Return a 10D observation with delayed pole information:
        [p0_pos, p0_vel, p1_pos, p1_vel, p2_pos, p2_vel, p3_pos, p3_vel, cart pos, cart vel].
        The pole information is delayed by one frame, while cart information is current.
        """
        joint_pos = self.cartpole.data.joint_pos
        joint_vel = self.cartpole.data.joint_vel

        # Get current pole positions and velocities
        current_p0_pos = joint_pos[:, self._pole_dof_idx[0]]
        current_p0_vel = joint_vel[:, self._pole_dof_idx[0]]
        current_p1_pos = joint_pos[:, self._pole01_dof_idx[0]]
        current_p1_vel = joint_vel[:, self._pole01_dof_idx[0]]
        current_p2_pos = joint_pos[:, self._pole02_dof_idx[0]]
        current_p2_vel = joint_vel[:, self._pole02_dof_idx[0]]
        current_p3_pos = joint_pos[:, self._pole03_dof_idx[0]]
        current_p3_vel = joint_vel[:, self._pole03_dof_idx[0]]

        # Normalize angles to [-π, π] range
        current_p0_pos = ((current_p0_pos + math.pi) % (2 * math.pi)) - math.pi
        current_p1_pos = ((current_p1_pos + math.pi) % (2 * math.pi)) - math.pi
        current_p2_pos = ((current_p2_pos + math.pi) % (2 * math.pi)) - math.pi
        current_p3_pos = ((current_p3_pos + math.pi) % (2 * math.pi)) - math.pi

        # Use previous pole data if available, otherwise use current
        p0_pos = self._prev_pole_pos if self._prev_pole_pos is not None else current_p0_pos
        p0_vel = self._prev_pole_vel if self._prev_pole_vel is not None else current_p0_vel
        p1_pos = self._prev_pole01_pos if self._prev_pole01_pos is not None else current_p1_pos
        p1_vel = self._prev_pole01_vel if self._prev_pole01_vel is not None else current_p1_vel
        p2_pos = self._prev_pole02_pos if self._prev_pole02_pos is not None else current_p2_pos
        p2_vel = self._prev_pole02_vel if self._prev_pole02_vel is not None else current_p2_vel
        p3_pos = self._prev_pole03_pos if self._prev_pole03_pos is not None else current_p3_pos
        p3_vel = self._prev_pole03_vel if self._prev_pole03_vel is not None else current_p3_vel

        # Store current pole data for next frame
        self._prev_pole_pos = current_p0_pos.clone()
        self._prev_pole_vel = current_p0_vel.clone()
        self._prev_pole01_pos = current_p1_pos.clone()
        self._prev_pole01_vel = current_p1_vel.clone()
        self._prev_pole02_pos = current_p2_pos.clone()
        self._prev_pole02_vel = current_p2_vel.clone()
        self._prev_pole03_pos = current_p3_pos.clone()
        self._prev_pole03_vel = current_p3_vel.clone()

        # Get current cart position and velocity (no delay)
        c_pos = joint_pos[:, self._cart_dof_idx[0]]
        c_vel = joint_vel[:, self._cart_dof_idx[0]]

        # Stack observations
        obs = torch.stack(
            (p0_pos, p0_vel, p1_pos, p1_vel, p2_pos, p2_vel, p3_pos, p3_vel, c_pos, c_vel),
            dim=-1
        )

        # Apply observation noise if enabled
        if hasattr(self.cfg, 'randomization_enable_observation_noise') and self.cfg.randomization_enable_observation_noise:
            # Create a noise config for the observation noise
            noise_cfg = GaussianNoiseCfg(mean=0.0, std=self.cfg.observation_noise_std, operation="add")
            # Apply noise directly using the gaussian_noise function
            obs = gaussian_noise(obs, noise_cfg)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Incorporate the 4th pole’s state into the reward function logic.
        Apply termination penalty when tip height is below threshold or cart is out of bounds.
        """
        joint_pos = self.cartpole.data.joint_pos
        joint_vel = self.cartpole.data.joint_vel

        p0_pos = joint_pos[:, self._pole_dof_idx[0]]
        p0_vel = joint_vel[:, self._pole_dof_idx[0]]

        p1_pos = joint_pos[:, self._pole01_dof_idx[0]]
        p1_vel = joint_vel[:, self._pole01_dof_idx[0]]

        p2_pos = joint_pos[:, self._pole02_dof_idx[0]]
        p2_vel = joint_vel[:, self._pole02_dof_idx[0]]

        p3_pos = joint_pos[:, self._pole03_dof_idx[0]]
        p3_vel = joint_vel[:, self._pole03_dof_idx[0]]

        c_pos = joint_pos[:, self._cart_dof_idx[0]]
        c_vel = joint_vel[:, self._cart_dof_idx[0]]

        # Calculate tip height for termination check
        L0 = L1 = L2 = L3 = 1.0
        tip_height = (
            L0 * torch.cos(p0_pos)
            + L1 * torch.cos(p0_pos + p1_pos)
            + L2 * torch.cos(p0_pos + p1_pos + p2_pos)
            + L3 * torch.cos(p0_pos + p1_pos + p2_pos + p3_pos)
        )

        # Check termination conditions
        cart_out_of_bounds = (torch.abs(c_pos) >= self.cfg.max_cart_pos)
        #below_balance_threshold = tip_height < self.balance_threshold['height']

        too_low_too_long = self.low_height_counter >= self.low_height_timeout_steps

        reset_terminated = cart_out_of_bounds | too_low_too_long# | below_balance_threshold

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_angle,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_cart_center,
            self.cfg.rew_scale_pole_straight,
            self.cfg.rew_scale_pole_upwards,
            self.cfg.rew_scale_exp_vel,
            self.cfg.vel_scale_factor,

            p0_pos,
            p0_vel,
            p1_pos,
            p1_vel,
            p2_pos,
            p2_vel,
            p3_pos,
            p3_vel,
            c_pos,
            c_vel,
            reset_terminated,

            self.cfg.bonus_cycle_length,
            self.balance_counter,
            self.cfg.rew_scale_bonus_first,
            self.cfg.rew_scale_bonus_second,
            self.cfg.rew_scale_bonus_third
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Check if we are done because:
         1) The cart is out of bounds, OR
         2) The final pole tip height is below balance threshold, OR
         3) Time is up
        """
        joint_pos = self.cartpole.data.joint_pos
        cart_x = joint_pos[:, self._cart_dof_idx[0]]

        # Angles to compute tip height
        p0 = joint_pos[:, self._pole_dof_idx[0]]
        p1 = joint_pos[:, self._pole01_dof_idx[0]]
        p2 = joint_pos[:, self._pole02_dof_idx[0]]
        p3 = joint_pos[:, self._pole03_dof_idx[0]]

        L0 = L1 = L2 = L3 = 1.0
        tip_height = (
            L0 * torch.cos(p0)
            + L1 * torch.cos(p0 + p1)
            + L2 * torch.cos(p0 + p1 + p2)
            + L3 * torch.cos(p0 + p1 + p2 + p3)
        )

        # Too low for too long check
        too_low_too_long = self.low_height_counter >= self.low_height_timeout_steps

        out_of_bounds = torch.abs(cart_x) > self.cfg.max_cart_pos
        #below_balance_threshold = tip_height < self.balance_threshold['height']

        terminated = out_of_bounds | too_low_too_long#  | below_balance_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset logic now randomizes the 4th pole's initial angle and velocity as well.
        Also resets the delayed pole information storage for the reset environments.
        """
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES

        super()._reset_idx(env_ids)


        # Reset the low height counter for these environments
        self.low_height_counter[env_ids] = 0

        # 1. Get default positions / velocities
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_vel = self.cartpole.data.default_joint_vel[env_ids] + sample_uniform(
            -0.1, 0.1, self.cartpole.data.default_joint_vel[env_ids].shape, joint_pos.device
        )

        # 2. Randomize each pole's angle
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pole01_dof_idx] += sample_uniform(
            self.cfg.initial_pole01_angle_range[0] * math.pi,
            self.cfg.initial_pole01_angle_range[1] * math.pi,
            joint_pos[:, self._pole01_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pole02_dof_idx] += sample_uniform(
            self.cfg.initial_pole02_angle_range[0] * math.pi,
            self.cfg.initial_pole02_angle_range[1] * math.pi,
            joint_pos[:, self._pole02_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pole03_dof_idx] += sample_uniform(
            self.cfg.initial_pole03_angle_range[0] * math.pi,
            self.cfg.initial_pole03_angle_range[1] * math.pi,
            joint_pos[:, self._pole03_dof_idx].shape,
            joint_pos.device,
        )

        # 3. Add small random initial velocities to all poles
        for dof_idx in [self._pole_dof_idx, self._pole01_dof_idx, self._pole02_dof_idx, self._pole03_dof_idx]:
            joint_vel[:, dof_idx] = sample_uniform(
                self.cfg.initial_velocity_range[0],
                self.cfg.initial_velocity_range[1],
                joint_vel[:, dof_idx].shape,
                joint_vel.device,
            )

        # 4. Shift root if needed
        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 5. Write the new joint states
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset step count
        self.episode_length_buf[env_ids] = 0
        
        # Also reset the delayed information for these environments
        if env_ids is self.cartpole._ALL_INDICES:
            # Reset all delayed information
            self._prev_pole_pos = None
            self._prev_pole_vel = None
            self._prev_pole01_pos = None
            self._prev_pole01_vel = None
            self._prev_pole02_pos = None
            self._prev_pole02_vel = None
            self._prev_pole03_pos = None
            self._prev_pole03_vel = None
        elif self._prev_pole_pos is not None:
            # For partial resets, only reset the specific environments
            # Get current joint positions and velocities
            current_joint_pos = self.cartpole.data.joint_pos
            current_joint_vel = self.cartpole.data.joint_vel
            
            # Update the affected indices with current values
            # (they'll be delayed by one frame naturally)
            self._prev_pole_pos[env_ids] = current_joint_pos[env_ids, self._pole_dof_idx[0]]
            self._prev_pole_vel[env_ids] = current_joint_vel[env_ids, self._pole_dof_idx[0]]
            self._prev_pole01_pos[env_ids] = current_joint_pos[env_ids, self._pole01_dof_idx[0]]
            self._prev_pole01_vel[env_ids] = current_joint_vel[env_ids, self._pole01_dof_idx[0]]
            self._prev_pole02_pos[env_ids] = current_joint_pos[env_ids, self._pole02_dof_idx[0]]
            self._prev_pole02_vel[env_ids] = current_joint_vel[env_ids, self._pole02_dof_idx[0]]
            self._prev_pole03_pos[env_ids] = current_joint_pos[env_ids, self._pole03_dof_idx[0]]
            self._prev_pole03_vel[env_ids] = current_joint_vel[env_ids, self._pole03_dof_idx[0]]


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_angle: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    rew_scale_cart_center: float,
    rew_scale_pole_straight: float,
    rew_scale_pole_upwards: float,
    rew_scale_exp_vel: float,
    vel_scale_factor: float,

    # Pole states
    p0_pos: torch.Tensor,
    p0_vel: torch.Tensor,
    p1_pos: torch.Tensor,
    p1_vel: torch.Tensor,
    p2_pos: torch.Tensor,
    p2_vel: torch.Tensor,
    p3_pos: torch.Tensor,
    p3_vel: torch.Tensor,

    # Cart states
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,

    # Termination
    reset_terminated: torch.Tensor,

    # Bonus cycle
    bonus_cycle_length: int,
    balance_counter: torch.Tensor,
    rew_scale_bonus_first: float,
    rew_scale_bonus_second: float,
    rew_scale_bonus_third: float
):
    """
    Reward function extended to include p3_pos/p3_vel in the same style.
    """

    # -- Example height-based reward for all four poles:
    L0 = L1 = L2 = L3 = 1.0
    # Summed angles for chain
    p1_abs = p0_pos + p1_pos
    p2_abs = p1_abs + p2_pos
    p3_abs = p2_abs + p3_pos
    y_tip = (
        L0 * torch.cos(p0_pos)
        + L1 * torch.cos(p1_abs)
        + L2 * torch.cos(p2_abs)
        + L3 * torch.cos(p3_abs)
    )
    max_height = L0 + L1 + L2 + L3  # 4.0
    height_reward = y_tip / max_height
    # Clip to avoid negative blow-ups if it flips
    height_reward = torch.clamp(height_reward, -1.00, 1.0)

    # Basic angle-based reward
    rew_angle = rew_scale_pole_angle * height_reward

    # "Alive" reward as long as not terminated
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # Termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # velocity penalties
    rew_cart_vel_penalty = rew_scale_cart_vel * torch.abs(cart_vel)
    total_pole_vel = torch.abs(p0_vel) + torch.abs(p1_vel) + torch.abs(p2_vel) + torch.abs(p3_vel)
    rew_pole_vel_penalty = rew_scale_pole_vel * total_pole_vel

    # cart center penalty (e.g. penalize distance from x=0)
    rew_center = -rew_scale_cart_center * torch.square(cart_pos)

    # Straightness reward: for example, average cos() of angles except p0 (which might be “down”)
    # or do your own logic. Here, let's just do a simple average of cos(p1), cos(p2), cos(p3):
    rew_straight = (torch.cos(p1_pos) + torch.cos(p2_pos) + torch.cos(p3_pos)) / 3.0
    rew_straight = rew_scale_pole_straight * rew_straight

    # Upwards bonus: require all 4 angles close to zero
    upright_angle_threshold = 0.1
    all_upright_mask = (
        (torch.abs(p0_pos) <= upright_angle_threshold)
        & (torch.abs(p1_pos) <= upright_angle_threshold)
        & (torch.abs(p2_pos) <= upright_angle_threshold)
        & (torch.abs(p3_pos) <= upright_angle_threshold)
    )
    # Some example scaling by speed factor
    total_speed = total_pole_vel / (vel_scale_factor + 1e-6)
    rew_upward_candidate = 1.0 - rew_scale_pole_upwards * total_speed
    rew_upward_candidate = torch.clamp(rew_upward_candidate, min=0.0)
    rew_upward = torch.zeros_like(p0_vel)
    rew_upward[all_upright_mask] = rew_upward_candidate[all_upright_mask]

    # Exponential velocity penalty (optional)
    # For example, we treat "uprightness" as a sum of cos(angles)
    uprightness_p0 = torch.cos(p0_pos)  # or negative if you want the first link reversed
    uprightness_p1 = torch.cos(p1_pos)
    uprightness_p2 = torch.cos(p2_pos)
    uprightness_p3 = torch.cos(p3_pos)
    total_uprightness = (uprightness_p0 + uprightness_p1 + uprightness_p2 + uprightness_p3) / 4.0
    exp_vel_penalty = rew_scale_exp_vel * torch.exp(total_uprightness * 2.0) * total_speed

    # Bonus cycle logic
    cycles = balance_counter // bonus_cycle_length
    bonus = torch.where(
        cycles == 1,
        rew_scale_bonus_first / float(bonus_cycle_length),
        torch.where(
            cycles == 2,
            rew_scale_bonus_second / float(bonus_cycle_length),
            torch.where(
                cycles >= 3,
                rew_scale_bonus_third / float(bonus_cycle_length),
                0.0
            )
        )
    )

    total_reward = (
        rew_alive
        + rew_termination
        + rew_angle
        + rew_center
        + rew_cart_vel_penalty
        + rew_pole_vel_penalty
        + rew_straight
        + rew_upward
        + exp_vel_penalty
        + bonus
    )
    return total_reward