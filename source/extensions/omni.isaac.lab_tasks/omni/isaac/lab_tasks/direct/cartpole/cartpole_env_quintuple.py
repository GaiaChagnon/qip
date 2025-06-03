# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# Import the articulation config for quadruple pole
from omni.isaac.lab_assets.cartpole_quintuple import CARTPOLE_CFG  # or wherever your .usd is located

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform

import logging
import sys
import gymnasium as gym

logger = logging.getLogger("cartpole_env_quintuple")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    file_handler = logging.FileHandler("cartpole_env_quintuple.log", mode="w")
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
    action_scale = 150.0
    action_space = 1
    # Now 4 poles => each has a position & velocity (8 values), plus cart pos + cart vel => 10
    observation_space = 12
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
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
    pole04_dof_name = "pole_to_pole_to_pole_to_pole_to_pole"        # 5th pole

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=5.0,
        replicate_physics=True
    )

    max_cart_pos = 3

    # Initial angles for each of the four poles
    initial_pole_angle_range = [0.0, 1.0]       # 1st pole
    initial_pole01_angle_range = [-0.04, 0.04]    # 2nd pole
    initial_pole02_angle_range = [-0.03, 0.03]    # 3rd pole
    initial_pole03_angle_range = [-0.02, 0.02]  # 4th pole
    initial_pole04_angle_range = [-0.01, 0.01]  # 5th pole

    # Small random initial velocity for all poles
    initial_velocity_range = [-0.005, 0.005]

    rew_scale_terminated = -3000.0
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


class CartpoleQuintupleEnv(DirectRLEnv):
    """
    Quadruple-pole cart environment, adapted from the triple-pole version.
    We now have four pendulum DOFs and incorporate them in the observation,
    reward, and reset logic.
    """

    cfg: CartpoleQuintupleEnvCfg

    def __init__(self, cfg: CartpoleQuaintupleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # DOF indices by name
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self._pole01_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole01_dof_name)
        self._pole02_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole02_dof_name)
        self._pole03_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole03_dof_name)
        self._pole04_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole04_dof_name)

        self.episode_info = {}
        self.action_scale = self.cfg.action_scale

        # For counting how many steps we've been "balanced"
        self.balance_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.previously_balanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # For four links, if each is length 1.0, the tip is near 4.0 in height at perfect upright
        self.balance_threshold = {
            'height': 4.65,  # slightly below 4.0 if "perfectly upright" is around 4.0
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
        3) Optionally logs debug info for a single environment (acceleration, velocity, etc.).
        """

        # 1) Convert the agent’s output into the scaled acceleration
        actions = actions.squeeze(-1)  # shape [num_envs]
        acceleration = torch.clamp(actions, -1.0, 1.0) * self.cfg.action_scale
        acceleration_2d = acceleration.unsqueeze(-1)  # shape [num_envs, 1]

        # 2) Read all joint positions for tip-height check
        pole1_angle = self.cartpole.data.joint_pos[:, self._pole_dof_idx[0]]
        pole2_angle = self.cartpole.data.joint_pos[:, self._pole01_dof_idx[0]]
        pole3_angle = self.cartpole.data.joint_pos[:, self._pole02_dof_idx[0]]
        pole4_angle = self.cartpole.data.joint_pos[:, self._pole03_dof_idx[0]]
        pole5_angle = self.cartpole.data.joint_pos[:, self._pole04_dof_idx[0]]

        L0 = L1 = L2 = L3 = L4 = 1.0
        tip_height = (
            L0 * torch.cos(pole1_angle)
            + L1 * torch.cos(pole1_angle + pole2_angle)
            + L2 * torch.cos(pole1_angle + pole2_angle + pole3_angle)
            + L3 * torch.cos(pole1_angle + pole2_angle + pole3_angle + pole4_angle)
            + L4 * torch.cos(pole1_angle + pole2_angle + pole3_angle + pole4_angle + pole5_angle)
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

        # 3) (Optional) Debug logging for one specific environment
        if self.cfg.debug:
            debug_env = self.cfg.debug_env_index
            if debug_env < self.num_envs:
                cart_pos_debug = self.cartpole.data.joint_pos[debug_env, self._cart_dof_idx[0]].item()
                cart_vel_debug = self.cartpole.data.joint_vel[debug_env, self._cart_dof_idx[0]].item()
                acc_debug = acceleration[debug_env].item()
                logger.debug(
                    f"[Env={debug_env}] "
                    f"Acc={acc_debug:.3f}, cartPos={cart_pos_debug:.3f}, "
                    f"cartVel={cart_vel_debug:.3f}"
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
        Return a 10D observation:
        [p0_pos, p0_vel, p1_pos, p1_vel, p2_pos, p2_vel, p3_pos, p3_vel, cart pos, cart vel].
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

        p4_pos = joint_pos[:, self._pole04_dof_idx[0]]
        p4_vel = joint_vel[:, self._pole04_dof_idx[0]]

        c_pos = joint_pos[:, self._cart_dof_idx[0]]
        c_vel = joint_vel[:, self._cart_dof_idx[0]]

        obs = torch.stack(
            (p0_pos, p0_vel, p1_pos, p1_vel, p2_pos, p2_vel, p3_pos, p3_vel, p4_pos, p4_vel, c_pos, c_vel),
            dim=-1
        )
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

        p4_pos = joint_pos[:, self._pole04_dof_idx[0]]
        p4_vel = joint_vel[:, self._pole04_dof_idx[0]]

        c_pos = joint_pos[:, self._cart_dof_idx[0]]
        c_vel = joint_vel[:, self._cart_dof_idx[0]]

        # Calculate tip height for termination check
        L0 = L1 = L2 = L3 = L4 = 1.0
        tip_height = (
            L0 * torch.cos(p0_pos)
            + L1 * torch.cos(p0_pos + p1_pos)
            + L2 * torch.cos(p0_pos + p1_pos + p2_pos)
            + L3 * torch.cos(p0_pos + p1_pos + p2_pos + p3_pos)
            + L4 * torch.cos(p0_pos + p1_pos + p2_pos + p3_pos + p4_pos)
        )

        # Check termination conditions
        cart_out_of_bounds = (torch.abs(c_pos) >= self.cfg.max_cart_pos)
        #below_balance_threshold = tip_height < self.balance_threshold['height']
        reset_terminated = cart_out_of_bounds# | below_balance_threshold

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
            p4_pos,
            p4_vel,
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
        p4 = joint_pos[:, self._pole04_dof_idx[0]]

        L0 = L1 = L2 = L3 = L4 = 1.0
        tip_height = (
            L0 * torch.cos(p0)
            + L1 * torch.cos(p0 + p1)
            + L2 * torch.cos(p0 + p1 + p2)
            + L3 * torch.cos(p0 + p1 + p2 + p3)
            + L4 * torch.cos(p0 + p1 + p2 + p3 + p4)
        )

        out_of_bounds = torch.abs(cart_x) > self.cfg.max_cart_pos
        #below_balance_threshold = tip_height < self.balance_threshold['height']

        terminated = out_of_bounds# | below_balance_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset logic now randomizes the 4th pole’s initial angle and velocity as well.
        """
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES

        super()._reset_idx(env_ids)

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

        joint_pos[:, self._pole04_dof_idx] += sample_uniform(
            self.cfg.initial_pole04_angle_range[0] * math.pi,
            self.cfg.initial_pole04_angle_range[1] * math.pi,
            joint_pos[:, self._pole04_dof_idx].shape,
            joint_pos.device,
        )

        # 3. Add small random initial velocities to all poles
        for dof_idx in [self._pole_dof_idx, self._pole01_dof_idx, self._pole02_dof_idx, self._pole03_dof_idx, self._pole04_dof_idx]:
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
    p4_pos: torch.Tensor,
    p4_vel: torch.Tensor,

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
    L0 = L1 = L2 = L3 = L4 = 1.0
    # Summed angles for chain
    p1_abs = p0_pos + p1_pos
    p2_abs = p1_abs + p2_pos
    p3_abs = p2_abs + p3_pos
    p4_abs = p3_abs + p4_pos
    y_tip = (
        L0 * torch.cos(p0_pos)
        + L1 * torch.cos(p1_abs)
        + L2 * torch.cos(p2_abs)
        + L3 * torch.cos(p3_abs)
        + L4 * torch.cos(p4_abs)
    )
    max_height = L0 + L1 + L2 + L3 + L4 # 5.0
    height_reward = y_tip / max_height
    # Clip to avoid negative blow-ups if it flips
    height_reward = torch.clamp(height_reward, -0.5, 1.0)

    # Basic angle-based reward
    rew_angle = rew_scale_pole_angle * height_reward

    # "Alive" reward as long as not terminated
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # Termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # velocity penalties
    rew_cart_vel_penalty = rew_scale_cart_vel * torch.abs(cart_vel)
    total_pole_vel = torch.abs(p0_vel) + torch.abs(p1_vel) + torch.abs(p2_vel) + torch.abs(p3_vel) + torch.abs(p4_vel)
    rew_pole_vel_penalty = rew_scale_pole_vel * total_pole_vel

    # cart center penalty (e.g. penalize distance from x=0)
    rew_center = -rew_scale_cart_center * torch.square(cart_pos)

    # Straightness reward: for example, average cos() of angles except p0 (which might be “down”)
    # or do your own logic. Here, let's just do a simple average of cos(p1), cos(p2), cos(p3):
    rew_straight = (torch.cos(p1_pos) + torch.cos(p2_pos) + torch.cos(p3_pos) + torch.cos(p4_pos)) / 4.0
    rew_straight = rew_scale_pole_straight * rew_straight

    # Upwards bonus: require all 4 angles close to zero
    upright_angle_threshold = 0.1
    all_upright_mask = (
        (torch.abs(p0_pos) <= upright_angle_threshold)
        & (torch.abs(p1_pos) <= upright_angle_threshold)
        & (torch.abs(p2_pos) <= upright_angle_threshold)
        & (torch.abs(p3_pos) <= upright_angle_threshold)
        & (torch.abs(p4_pos) <= upright_angle_threshold)
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
    uprightness_p3 = torch.cos(p4_pos)
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