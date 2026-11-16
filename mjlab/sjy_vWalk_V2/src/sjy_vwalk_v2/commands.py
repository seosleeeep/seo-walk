"""User's weighted staged command sampler, adapted to mjlab's reset API."""
from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

def sample_uniform(low, high, shape, device):
    return torch.empty(shape, device=device).uniform_(low, high)

class INHAWalkCommand(CommandTerm):
    cfg: INHAWalkCommandCfg

    def __init__(self, cfg: INHAWalkCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 10, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def gait_frequency(self) -> torch.Tensor:
        return self._command[:, 3]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(self.cfg.play_velocity_command, device=self.device)
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
            return
        vr = self.cfg.velocity_ranges
        self._command[env_ids, 0] = sample_uniform(*vr.lin_vel_x, (len(env_ids),), device=self.device)
        self._command[env_ids, 1] = sample_uniform(*vr.lin_vel_y, (len(env_ids),), device=self.device)
        self._command[env_ids, 2] = sample_uniform(*vr.ang_vel_yaw, (len(env_ids),), device=self.device)
        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            self._command[env_ids, 3] = sample_uniform(*pr.gait_frequency, (len(env_ids),), device=self.device)
            self._command[env_ids, 4] = sample_uniform(*pr.foot_yaw_L, (len(env_ids),), device=self.device)
            self._command[env_ids, 5] = sample_uniform(*pr.foot_yaw_R, (len(env_ids),), device=self.device)
            self._command[env_ids, 6] = sample_uniform(*pr.body_pitch_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 7] = sample_uniform(*pr.body_roll_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 8] = sample_uniform(*pr.feet_offset_x_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 9] = sample_uniform(*pr.feet_offset_y_target, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_still]]
            self._command[still_ids, 0:4] = 0.0
            self.gait_process[still_ids] = 0.0

    def _update_command(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.gait_process[env_ids] = torch.fmod(self.gait_process[env_ids] + self._env.step_dt * self.gait_frequency[env_ids], 1.0)

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

class StagedCombinationWalkCommand(INHAWalkCommand):
    """Sample pure commands first, then expand to every non-empty axis subset."""
    cfg: StagedCombinationWalkCommandCfg
    _PURE_MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    _ALL_MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(self.cfg.play_velocity_command, device=self.device)
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
            if max(abs(x) for x in self.cfg.play_velocity_command) < 0.05:
                self._command[env_ids, 3] = 0.0
                self.gait_process[env_ids] = 0.0
            return
        iteration = int(self._env.common_step_counter) // max(self.cfg.num_steps_per_iteration, 1)
        modes = self._PURE_MODES if iteration < self.cfg.pure_command_iterations else self._ALL_MODES
        mode_table = torch.tensor(modes, dtype=self._command.dtype, device=self.device)
        mode_ids = torch.randint(len(modes), (len(env_ids),), device=self.device)
        active_axes = mode_table[mode_ids]
        vr = self.cfg.velocity_ranges
        sampled = torch.stack((sample_uniform(*vr.lin_vel_x, (len(env_ids),), device=self.device), sample_uniform(*vr.lin_vel_y, (len(env_ids),), device=self.device), sample_uniform(*vr.ang_vel_yaw, (len(env_ids),), device=self.device)), dim=1)
        self._command[env_ids, 0:3] = sampled * active_axes
        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            ranges = (pr.gait_frequency, pr.foot_yaw_L, pr.foot_yaw_R, pr.body_pitch_target, pr.body_roll_target, pr.feet_offset_x_target, pr.feet_offset_y_target)
            for command_index, value_range in enumerate(ranges, start=3):
                self._command[env_ids, command_index] = sample_uniform(*value_range, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_still]]
            self._command[still_ids, 0:4] = 0.0
            self.gait_process[still_ids] = 0.0

class WeightedStagedCombinationWalkCommand(StagedCombinationWalkCommand):
    """Keep the pure stage, then curriculum the 1/2/3-axis topology ratios.

    ``pure_command_iterations`` is measured from training iteration zero.  The
    topology curriculum begins only after that pure stage, and its stage
    boundaries are measured locally from the all-mode transition.
    """
    cfg: WeightedStagedCombinationWalkCommandCfg
    _DOUBLE_MODES = ((1, 1, 0), (1, 0, 1), (0, 1, 1))

    def _topology_probabilities(self, all_mode_iteration: int) -> tuple[float, float, float]:
        if all_mode_iteration < self.cfg.topology_stage_1_iterations:
            return (0.3, 0.4, 0.3)
        if all_mode_iteration < self.cfg.topology_stage_2_iterations:
            return (0.2, 0.4, 0.4)
        if all_mode_iteration < self.cfg.topology_stage_3_iterations:
            return (0.15, 0.3, 0.55)
        return (0.1, 0.3, 0.6)

    def _sample_active_axes(self, count: int, iteration: int) -> torch.Tensor:
        pure_modes = self._command.new_tensor(self._PURE_MODES)
        if iteration < self.cfg.pure_command_iterations:
            mode_ids = torch.randint(len(self._PURE_MODES), (count,), device=self.device)
            return pure_modes[mode_ids]
        all_mode_iteration = iteration - self.cfg.pure_command_iterations
        one_axis_p, two_axis_p, _ = self._topology_probabilities(all_mode_iteration)
        topology_draw = torch.rand(count, device=self.device)
        one_axis_mask = topology_draw < one_axis_p
        two_axis_mask = (topology_draw >= one_axis_p) & (topology_draw < one_axis_p + two_axis_p)
        three_axis_mask = ~(one_axis_mask | two_axis_mask)
        active_axes = torch.zeros(count, 3, dtype=self._command.dtype, device=self.device)
        pure_choice = torch.randint(len(self._PURE_MODES), (count,), device=self.device)
        double_modes = self._command.new_tensor(self._DOUBLE_MODES)
        double_choice = torch.randint(len(self._DOUBLE_MODES), (count,), device=self.device)
        active_axes[one_axis_mask] = pure_modes[pure_choice[one_axis_mask]]
        active_axes[two_axis_mask] = double_modes[double_choice[two_axis_mask]]
        active_axes[three_axis_mask] = 1.0
        return active_axes

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.play:
            self._command[env_ids, 0:3] = self._command.new_tensor(self.cfg.play_velocity_command)
            self._command[env_ids, 3:10] = self._command.new_tensor(self.cfg.default_internal_parameters)
            if max(abs(x) for x in self.cfg.play_velocity_command) < 0.05:
                self._command[env_ids, 3] = 0.0
                self.gait_process[env_ids] = 0.0
            return
        iteration = int(self._env.common_step_counter) // max(self.cfg.num_steps_per_iteration, 1)
        active_axes = self._sample_active_axes(len(env_ids), iteration)
        velocity_ranges = self.cfg.velocity_ranges
        sampled_velocity = torch.stack((sample_uniform(*velocity_ranges.lin_vel_x, (len(env_ids),), device=self.device), sample_uniform(*velocity_ranges.lin_vel_y, (len(env_ids),), device=self.device), sample_uniform(*velocity_ranges.ang_vel_yaw, (len(env_ids),), device=self.device)), dim=1)
        self._command[env_ids, 0:3] = sampled_velocity * active_axes
        if self.cfg.sample_internal_parameters:
            parameter_ranges = self.cfg.parameter_ranges
            ranges = (parameter_ranges.gait_frequency, parameter_ranges.foot_yaw_L, parameter_ranges.foot_yaw_R, parameter_ranges.body_pitch_target, parameter_ranges.body_roll_target, parameter_ranges.feet_offset_x_target, parameter_ranges.feet_offset_y_target)
            for command_index, value_range in enumerate(ranges, start=3):
                self._command[env_ids, command_index] = sample_uniform(*value_range, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = self._command.new_tensor(self.cfg.default_internal_parameters)
        # Reserve half of moving samples for unassisted forward tracking.
        forward = torch.rand(len(env_ids), device=self.device) < self.cfg.forward_proportion
        forward_ids = env_ids[forward]
        self._command[forward_ids, 0] = sample_uniform(
            .2, velocity_ranges.lin_vel_x[1], (len(forward_ids),), self.device)
        self._command[forward_ids, 1:3] = 0.
        # Bernoulli sampling also works when only one environment resamples.
        still = torch.rand(len(env_ids), device=self.device) < self.cfg.still_proportion
        still_ids = env_ids[still]
        self._command[still_ids, 0:4] = 0.
        self.gait_process[still_ids] = 0.

@dataclass
class VelocityRanges:
    lin_vel_x: tuple[float,float] = (-1.5, 3.0)
    lin_vel_y: tuple[float,float] = (-1.5, 1.5)
    ang_vel_yaw: tuple[float,float] = (-1.6, 1.6)

class WaypointWalkCommand(WeightedStagedCombinationWalkCommand):
    """Velocity training; optional waypoint controller only for play evaluation."""
    def read_planar_pose(self, ids):
        """Return XY in meters and yaw in radians in one fixed frame (future odom input)."""
        data = self.robot.data
        w, x, y, z = data.root_link_quat_w[ids].unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return data.root_link_pos_w[ids, :2], yaw

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids = torch.arange(self.num_envs, device=self.device)
        self.waypoint = torch.zeros_like(ids, dtype=torch.bool) if not cfg.play else torch.full_like(ids, cfg.distance is not None, dtype=torch.bool)
        self.goal = self._command.new_zeros(self.num_envs, 2)
        self.heading = self._command.new_zeros(self.num_envs)
        self.speed = torch.zeros_like(self.heading)
        self.arrived = torch.zeros_like(self.waypoint)
        self.pending = self.waypoint.clone()

    def _resample_command(self, env_ids):
        phase = self.gait_process[env_ids].clone()
        super()._resample_command(env_ids)
        self.gait_process[env_ids] = torch.where(self.waypoint[env_ids],phase,self.gait_process[env_ids])
        self._command[env_ids,:4] = torch.where(self.pending[env_ids,None],0.,self._command[env_ids,:4])

    def _update_command(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        pos, yaw = self.read_planar_pose(ids)
        c,s = yaw.cos(), yaw.sin()
        new = self.pending[ids]
        # Sample only at episode boundaries; regular velocity resampling never moves the goal.
        if new.any():
            idx = torch.arange(self.num_envs, device=self.device)[ids][new]
            n = len(idx)
            d = self.cfg.distance if self.cfg.play else sample_uniform(1.,4.,(n,),self.device)
            self.goal[idx,0] = pos[new,0] + d*c[new]
            self.goal[idx,1] = pos[new,1] + d*s[new]
            self.heading[idx] = yaw[new]
            self.speed[idx] = self.cfg.play_velocity_command[0] if self.cfg.play else sample_uniform(.3,3.,(n,),self.device)
            self.pending[idx] = False
            self.arrived[idx] = False
        dx,dy = (self.goal[ids]-pos).unbind(-1)
        err = self.heading[ids]-yaw
        angle = torch.atan2(err.sin(),err.cos())
        stop = self.arrived[ids] | ((dx.square()+dy.square() < .10**2) & (angle.abs()<.09))
        self.arrived[ids] = stop & self.waypoint[ids]
        vx = torch.minimum((1.4*(c*dx+s*dy)).clamp_min(-.4),self.speed[ids])
        vy = (1.4*(-s*dx+c*dy)).clamp(-.6,.6)
        out = torch.stack((vx,vy,(2.*angle).clamp(-1.,1.),torch.full_like(vx,self.cfg.default_internal_parameters[0])),dim=-1)
        mask = self.waypoint[ids]
        self._command[ids,:4] = torch.where(mask[:,None],torch.where(stop[:,None],0.,out),self._command[ids,:4])
        self.gait_process[ids] = torch.where(mask & stop,0.,self.gait_process[ids])
        super()._update_command(ids)

@dataclass(kw_only=True)
class WalkCommandCfg(CommandTermCfg):
    asset_name: str = "robot"
    velocity_ranges: VelocityRanges = field(default_factory=VelocityRanges)
    still_proportion: float = 0.1
    forward_proportion: float = 0.5
    sample_internal_parameters: bool = False
    default_internal_parameters: tuple = (1.5,0.,0.,0.,0.,0.,0.)
    pure_command_iterations: int = 500
    topology_stage_1_iterations: int = 500
    topology_stage_2_iterations: int = 500
    topology_stage_3_iterations: int = 500
    num_steps_per_iteration: int = 24
    play: bool = False
    distance: float | None = None
    play_velocity_command: tuple[float,float,float] = (0.,0.,0.)

    def build(self, env):
        from .curriculum_command import CurriculumWalkCommand
        return CurriculumWalkCommand(self, env)
