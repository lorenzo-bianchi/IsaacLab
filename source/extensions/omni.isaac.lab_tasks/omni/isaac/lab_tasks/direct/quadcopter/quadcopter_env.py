# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

from matplotlib import pyplot as plt
from collections import deque

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0             # episode_length = episode_length_s / dt / decimation
    decimation = 2
    action_space = 4
    observation_space = (
        3 +     # linear velocity
        3 +     # angular velocity
        3 +     # relative desired position
        9 +     # attitude matrix
        4 +     # last actions
        1       # absolute height
    )
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    rewards = {}

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get train/test mode
        if self.num_envs > 10:
            self.is_train = True
        else:
            self.is_train = False

        if len(cfg.rewards) > 0:
            self.rew = cfg.rewards
        elif self.is_train:
            raise ValueError("rewards not provided")

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.last_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.beta = 0.9
        self.min_altitude = 0.1
        self.max_altitude = 2.0
        self.max_time_on_ground = 2.0
        self.reset_mode = "alt_att" # "alt_no_att", "alt_att", "ground"

        self.last_yaw = 0.0
        self.n_laps = torch.zeros(self.num_envs, device=self.device)
        self.prob_change = 0.5
        self.proximity_threshold = 0.1
        self.wait_time_s = 1.0
        self.t_previous = torch.zeros(self.num_envs, device=self.device)

        self.episode_length_buf_zero = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.min_roll_pitch = -torch.pi / 4.0
        self.max_roll_pitch =  torch.pi /4.0
        self.min_yaw = -torch.pi
        self.max_yaw =  torch.pi
        self.min_lin_vel_xy = -0.2
        self.max_lin_vel_xy =  0.2
        self.min_lin_vel_z = -0.1
        self.max_lin_vel_z =  0.1
        self.min_ang_vel = -0.1
        self.max_ang_vel =  0.1

        if not self.is_train:
            self.change_setpoint = True
            if self.change_setpoint:
                cfg.episode_length_s = 20.0
            else:
                cfg.episode_length_s = 20.0
            self.draw_plots = True
            self.max_len_deque = 100
            self.roll_history = deque(maxlen=self.max_len_deque)
            self.pitch_history = deque(maxlen=self.max_len_deque)
            self.yaw_history = deque(maxlen=self.max_len_deque)
            self.actions_history = deque(maxlen=self.max_len_deque)
            self.n_steps = 0
            self.rpy_fig, self.rpy_axes = plt.subplots(4, 1, figsize=(10, 8))
            self.roll_line, = self.rpy_axes[0].plot([], [], 'r', label="Roll")
            self.pitch_line, = self.rpy_axes[1].plot([], [], 'g', label="Pitch")
            self.yaw_line, = self.rpy_axes[2].plot([], [], 'b', label="Yaw")
            self.actions_lines = [self.rpy_axes[3].plot([], [], label=f"{legend}")[0] for legend in ["Thrust", "Roll rate", "Pitch rate", "Yaw rate"]]

            if self.num_envs > 1:
                self.draw_plots = False
                plt.close(self.rpy_fig)

            # Configure subplots
            for ax, title in zip(self.rpy_axes, ["Roll History", "Pitch History", "Yaw History", "Actions History"]):
                ax.set_title(title)
                ax.set_xlabel("Time Step")
                if any(angle in title for angle in ["Roll", "Pitch", "Yaw"]):
                    ax.set_ylabel("Angle (°)")
                elif title == "Actions History":
                    ax.set_ylabel("Action")
                ax.legend(loc="upper left")
                ax.grid(True)

            plt.tight_layout()
            plt.ion()  # interactive mode

        # Logging
        # get keys from self.rew.keys() except for death_cost and remove the _reward_scale suffix
        if self.is_train:
            keys = [key.split("_reward_scale")[0] for key in self.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in keys
            }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._actions = self.beta * self._actions + (1 - self.beta) * self.last_actions
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w)

        quat_w = self._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self._robot.data.root_link_state_w[:, 3].unsqueeze(1),
                desired_pos_b,
                attitude_mat.view(attitude_mat.shape[0], -1),
                self._robot.data.root_com_lin_vel_b,
                self._robot.data.root_com_ang_vel_b,
                self.last_actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.last_yaw
        self.n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.unwrapped_yaw = yaw_w + 2 * np.pi * self.n_laps
        self.last_yaw = yaw_w

        if not self.is_train:
            if self.draw_plots:
                # RPY plots
                roll_w = wrap_to_pi(rpy[0])
                pitch_w = wrap_to_pi(rpy[1])

                self.roll_history.append(roll_w * 180.0 / np.pi)
                self.pitch_history.append(pitch_w * 180.0 / np.pi)
                self.yaw_history.append(self.unwrapped_yaw * 180.0 / np.pi)
                self.actions_history.append(self._actions.squeeze(0).cpu().numpy())

                self.n_steps += 1
                if self.n_steps >= self.max_len_deque:
                    steps = np.arange(self.n_steps - self.max_len_deque, self.n_steps)
                else:
                    steps = np.arange(self.n_steps)

                self.roll_line.set_data(steps, self.roll_history)
                self.pitch_line.set_data(steps, self.pitch_history)
                self.yaw_line.set_data(steps, self.yaw_history)

                for i in range(self.cfg.action_space):
                    self.actions_lines[i].set_data(steps, np.array(self.actions_history)[:, i])

                for ax in self.rpy_axes:
                    ax.relim()
                    ax.autoscale_view()

                plt.draw()
                plt.pause(0.001)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation  # updated at each step
        close_to_goal = (distance_to_goal < self.proximity_threshold).to(self.device)
        time_cond = (episode_time - self.t_previous) >= self.wait_time_s
        give_reward = torch.logical_and(close_to_goal, time_cond)
        ids = torch.where(give_reward)[0]

        if self.is_train:
            lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)

            approaching = torch.relu(self.last_distance_to_goal - distance_to_goal)
            # approaching = self.closest_distance_to_goal - distance_to_goal
            # self.closest_distance_to_goal = torch.minimum(self.closest_distance_to_goal, distance_to_goal)
            # approaching = torch.clip(approaching, 0, 100)
            convergence = 1 - torch.tanh(distance_to_goal / 0.04)

            yaw_w_mapped = torch.exp(-10.0 * torch.abs(self.unwrapped_yaw))

            cmd_smoothness = torch.sum(torch.square(self._actions - self.last_actions), dim=1)
            cmd_body_rates_smoothness = torch.sum(torch.square(self._actions[:, 1:]), dim=1)

            rewards = {
                "lin_vel": lin_vel * self.rew['lin_vel_reward_scale'] * self.step_dt,
                "ang_vel": ang_vel * self.rew['ang_vel_reward_scale'] * self.step_dt,

                "approaching_goal": approaching * self.rew['approaching_goal_reward_scale'] * self.step_dt,
                "convergence_goal": convergence * self.rew['convergence_goal_reward_scale'] * self.step_dt,

                "yaw": yaw_w_mapped * self.rew['yaw_reward_scale'] * self.step_dt,

                "cmd_smoothness": cmd_smoothness * self.rew['cmd_smoothness_reward_scale'] * self.step_dt,
                "cmd_body_rates": cmd_body_rates_smoothness * self.rew['cmd_body_rates_reward_scale'] * self.step_dt,

                "new_goal": give_reward * self.rew['new_goal_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.rew['death_cost'], reward)

            self.last_distance_to_goal = distance_to_goal.clone()

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        self.last_actions = self._actions.clone()

        self.t_previous[ids] = episode_time[ids]
        change_setpoint_mask = (torch.rand(len(ids), device=self.device) < self.prob_change)
        ids_to_change = ids[change_setpoint_mask]
        if len(ids_to_change) > 0:
            self._desired_pos_w[ids_to_change, :2] = torch.zeros_like(self._desired_pos_w[ids_to_change, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[ids_to_change, :2] += self._terrain.env_origins[ids_to_change, :2]
            self._desired_pos_w[ids_to_change, 2] = torch.zeros_like(self._desired_pos_w[ids_to_change, 2]).uniform_(0.5, 1.5)
            self.t_previous[ids_to_change] = 0.0

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cond_h_min_time = torch.logical_and(
            self._robot.data.root_link_pos_w[:, 2] < self.min_altitude, \
            (self.episode_length_buf - self.episode_length_buf_zero) * self.cfg.sim.dt * self.cfg.decimation > self.max_time_on_ground
        )
        died = torch.logical_or(cond_h_min_time, self._robot.data.root_link_pos_w[:, 2] > self.max_altitude)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.is_train:
            # Logging
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
            ).mean()
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.extras["log"] = dict()
            self.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
            self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.episode_length_buf_zero = self.episode_length_buf.clone()

        self._actions[env_ids] = 0.0

        # Sample new desired position
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Reset joints state
        joint_pos = self._robot.data.default_joint_pos[env_ids]     # not important
        joint_vel = self._robot.data.default_joint_vel[env_ids]     #
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset robots state
        default_root_state = self._robot.data.default_root_state[env_ids]   # [pos, quat, lin_vel, ang_vel] in local environment frame. Shape is (num_instances, 13)
        if self.reset_mode == "alt_no_att":
            pass
        elif self.reset_mode == "alt_att":
            pos = default_root_state[:, :3]
            # Randomize other values
            quat = quat_from_euler_xyz(
                torch.FloatTensor(n_reset).uniform_(self.min_roll_pitch, self.max_roll_pitch),
                torch.FloatTensor(n_reset).uniform_(self.min_roll_pitch, self.max_roll_pitch),
                torch.FloatTensor(n_reset).uniform_(self.min_yaw, self.max_yaw)
            )
            lin_vel = torch.stack([
                torch.FloatTensor(n_reset).uniform_(self.min_lin_vel_xy, self.max_lin_vel_xy),
                torch.FloatTensor(n_reset).uniform_(self.min_lin_vel_xy, self.max_lin_vel_xy),
                torch.FloatTensor(n_reset).uniform_(self.min_lin_vel_z, self.max_lin_vel_z)
            ], dim=1)
            ang_vel = torch.FloatTensor(n_reset, 3).uniform_(self.min_ang_vel, self.max_ang_vel)
            default_root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=1)
        elif self.reset_mode == "ground":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, 2] = 0.0
        else:
            raise ValueError(f"Unknown reset mode: {self.reset_mode}")
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.n_laps[env_ids] = 0
        self.t_previous[env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)