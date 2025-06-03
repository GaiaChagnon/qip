# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

# Import the articulation config for the double-pole cart
from omni.isaac.lab_assets.cartpole_double import CARTPOLE_CFG

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

# Setup logger at DEBUG level
logger = logging.getLogger("cartpole_env_double")
logger.setLevel(logging.DEBUG)  # Changed to DEBUG for detailed logging


if not logger.handlers:

    file_handler = logging.FileHandler("cartpole_env_double.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    #Write to real stdout
    stdout_handler = logging.StreamHandler(sys.__stdout__)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    """
    Example config for a double-pole cart environment using an older lab Articulation.
    Now we remove the explicit 'cart_mass' and let PhysX handle the force internally
    by using velocity control on the prismatic DOF.
    """

    # == Basic Env Settings ==
    decimation = 2
    episode_length_s = 1.0
    action_scale = 50.0  # up to ±20 m/s^2 if raw action is ±1
    action_space = 1
    observation_space = 6
    state_space = 0

    # == Simulation Settings ==
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation
    )

    # == Robot / Articulation ==
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pole01_dof_name = "pole_to_pole"

    # == Scene ==
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=5.0,
        replicate_physics=True
    )

    # == Reset/Bounds ==
    max_cart_pos = 3
    initial_pole_angle_range = [-0.05, 0.05]
    initial_pole01_angle_range = [-0.03, 0.03]

    rew_scale_terminated = -500.0  # a fairly strong penalty if the cart goes OOB or agent ends early
    bonus_cycle_length: int = 80

    # Configurable values with defaults
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

    logger.debug(f"CartpoleDoubleEnv: Using rew_scale_alive = {rew_scale_alive}")




class CartpoleDoubleEnv(DirectRLEnv):
    """
    Double-pole cart environment using older-lab style articulation.
    Now the prismatic DOF is driven via velocity control, and the RL action
    is interpreted as an *acceleration* that updates the velocity target each step.
    """
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find the DOF indices by name
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self._pole01_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole01_dof_name)

        self.episode_info = {}

        # We'll interpret the raw RL action as "desired acceleration" (±action_scale).
        self.action_scale = self.cfg.action_scale

        self.balance_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.previously_balanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Track state of each env

        self.balance_threshold = {
            'height': 1.85,  # How close to vertical the poles need to be (in meters)
        }
        self.min_steps_balanced = 240  # Number of steps needed for successful balance
        self.total_successful_envs = 0  # Track total number of successful environments

    def _setup_scene(self):
        """
        Create the articulation from config, ground plane, etc.
        """
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["cartpole"] = self.cartpole

        # optional dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.squeeze(-1)
        force = torch.clamp(actions, -1.0, 1.0) * self.action_scale
        force_2d = force.unsqueeze(-1)  # Make sure shape is [num_envs, 1]

        joint_pos = self.cartpole.data.joint_pos
        joint_vel = self.cartpole.data.joint_vel

        pole1_angle = self.cartpole.data.joint_pos[:, self._pole_dof_idx[0]]
        pole2_angle = self.cartpole.data.joint_pos[:, self._pole01_dof_idx[0]]
        pole1_vel = self.cartpole.data.joint_vel[:, self._pole_dof_idx[0]]
        pole2_vel = self.cartpole.data.joint_vel[:, self._pole01_dof_idx[0]]
        
        L0 = 1.0
        L1 = 1.0
        tip_height = L0 * torch.cos(pole1_angle) + L1 * torch.cos(pole1_angle + pole2_angle)
        
        # Check which environments have a tip height above the threshold
        # (and optionally that the pole angular velocities are small)
        balanced = (
            (tip_height > self.balance_threshold['height'])
        )
        
        # Update consecutive balance counter
        self.balance_counter[balanced] += 1
        self.balance_counter[~balanced] = 0
        
        # Check which envs are currently successful
        currently_successful = self.balance_counter >= self.min_steps_balanced
        
        # Count new successes (environments that just became successful)
        new_successes = currently_successful & ~self.previously_balanced
        self.total_successful_envs += new_successes.sum().item()
        
        # Update our memory of which envs were successful
        self.previously_balanced = currently_successful
        self.episode_info["final_balance_count"] = self.total_successful_envs


        # Now we pass a shape [num_envs, 1] target 
        # to match joint_ids=[0].
        # Directly set the joint effort target (i.e. the force to apply)
        self.cartpole.set_joint_effort_target(force_2d, joint_ids=self._cart_dof_idx)


    def _apply_action(self) -> None:
        self.cartpole.write_data_to_sim()
        pass

    def _get_observations(self) -> dict:
        """
        For older-lab code, we read the actual joint states from `self.cartpole.data`.
        We'll create a 6D observation per env:
        [pole1 pos, pole1 vel, pole0 pos, pole0 vel, cart pos, cart vel].
        """
        joint_pos = self.cartpole.data.joint_pos  # shape [num_envs, num_joints]
        joint_vel = self.cartpole.data.joint_vel  # shape [num_envs, num_joints]

        # index them:
        p1_pos = joint_pos[:, self._pole01_dof_idx[0]]
        p1_vel = joint_vel[:, self._pole01_dof_idx[0]]
        p0_pos = joint_pos[:, self._pole_dof_idx[0]]
        p0_vel = joint_vel[:, self._pole_dof_idx[0]]
        c_pos  = joint_pos[:, self._cart_dof_idx[0]]
        c_vel  = joint_vel[:, self._cart_dof_idx[0]]
        #print("Positions---")
        #print("P1", p1_pos, "P0", p0_pos, "Cart", c_pos)
        #logger.debug(f"Pole0 - Pos: {p0_pos[0]:.3f}, Vel: {p0_vel[0]:.3f}")
        # stack them into [num_envs, 6]
        obs = torch.stack((p0_pos, p0_vel, p1_pos, p1_vel, c_pos, c_vel), dim=-1)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute a per-env reward using joint positions & velocities.
        """
        joint_pos = self.cartpole.data.joint_pos
        joint_vel = self.cartpole.data.joint_vel

        p0_pos = joint_pos[:, self._pole_dof_idx[0]]
        p0_vel = joint_vel[:, self._pole_dof_idx[0]]
        p1_pos = joint_pos[:, self._pole01_dof_idx[0]]
        p1_vel = joint_vel[:, self._pole01_dof_idx[0]]
        c_pos  = joint_pos[:, self._cart_dof_idx[0]]
        c_vel  = joint_vel[:, self._cart_dof_idx[0]]

        cart_x = c_pos
        reset_terminated = (torch.abs(cart_x) >= self.cfg.max_cart_pos)

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
        Check if we are done because the cart is out of bounds, or time is up.
        """
        joint_pos = self.cartpole.data.joint_pos
        cart_x = joint_pos[:, self._cart_dof_idx[0]]

        out_of_bounds = torch.abs(cart_x) > self.cfg.max_cart_pos

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset selected environments. Also reset the velocity target to 0.0.
        """
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        # 1. retrieve default positions/velocities
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        # 2. randomize your poles
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

        # 3. shift root if needed
        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # 4. write the new joint states
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

    p0_pos: torch.Tensor,
    pole0_vel: torch.Tensor,
    p1_pos: torch.Tensor,
    pole1_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,

    # --- New bonus cycle parameters ---
    bonus_cycle_length: int,
    balance_counter: torch.Tensor,
    rew_scale_bonus_first: float,
    rew_scale_bonus_second: float,
    rew_scale_bonus_third: float
):
    #Pole straightness : 
    rew_straight = torch.cos(p1_pos) * rew_scale_pole_straight

    upright_angle_threshold = 0.1

    p1_abs = p0_pos + p1_pos

    # 0) Prepare a baseline rew_upward (tensor of zeros)
    rew_upward = torch.zeros_like(pole0_vel)  # shape [N], same device/dtype as velocities
    
    # Normalize velocities
    total_speed = (torch.abs(pole0_vel) + torch.abs(pole1_vel)) / vel_scale_factor

    # For rew_upward calculation
    both_upright_mask = (
        (torch.abs(p0_pos) <= upright_angle_threshold)
        &
        (torch.abs(p1_pos) <= upright_angle_threshold)
    )
    
    rew_upward_candidate = 1.0 - rew_scale_pole_upwards * total_speed
    rew_upward_candidate = torch.clamp(rew_upward_candidate, min=0.0)
    rew_upward[both_upright_mask] = rew_upward_candidate[both_upright_mask]

    # For exponential velocity penalty
    uprightness_p0 = -torch.cos(p0_pos)
    uprightness_p1 = torch.cos(p1_pos)
    total_uprightness = (uprightness_p0 + uprightness_p1) / 2.0

    exp_vel_penalty = rew_scale_exp_vel * torch.exp(total_uprightness * 2) * total_speed


    #Compute tip height
    L0 = 1.0  # or your actual link length
    L1 = 1.0  # or your actual link length
    y_tip = L0 * torch.cos(p0_pos) + L1 * torch.cos(p1_abs)
    max_height = L0 + L1
    height_reward = y_tip / max_height

    height_reward = torch.clamp(height_reward, 0.0, 1.0)

    # Then scale by your desired factor
    rew_angle = rew_scale_pole_angle * height_reward

    # standard living + termination
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())


    # Scale the penalty so that when distance_into_edge equals edge_margin, the penalty equals rew_scale_terminated.
    rew_termination = rew_scale_terminated * reset_terminated.float()

    # velocity penalties
    rew_cart_vel_penalty = rew_scale_cart_vel * torch.abs(cart_vel)
    total_pole_vel = torch.abs(pole0_vel) + torch.abs(pole1_vel)
    rew_pole_vel_penalty = rew_scale_pole_vel * total_pole_vel

    # cart center reward
    rew_center = -rew_scale_cart_center * torch.square(cart_pos)

    # --- Bonus Cycle Logic ---
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