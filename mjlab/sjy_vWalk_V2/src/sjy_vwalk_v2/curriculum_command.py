"""Only changes the positive vx sampling ceiling and measures promotion gates."""
import json
import math
import torch
from .commands import WaypointWalkCommand
from .speed_curriculum import SpeedGate


class CurriculumWalkCommand(WaypointWalkCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.speed_gate = SpeedGate()
        if not cfg.play:
            self.cfg.velocity_ranges.lin_vel_x = (self.cfg.velocity_ranges.lin_vel_x[0], self.speed_gate.cap)
        self.segment_age = self._command.new_zeros(self.num_envs)
        self.segment_samples = self._command.new_zeros(self.num_envs)
        self.segment_good = self._command.new_zeros(self.num_envs)
        self.segment_eligible = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.segment_origin = self._command.new_zeros(self.num_envs, 2)
        self.segment_yaw = self._command.new_zeros(self.num_envs)
        for key in ('speed_cap_mps', 'gate_success_rate', 'gate_stage'):
            self.metrics[key] = self._command.new_zeros(self.num_envs)

    def _resample_command(self, env_ids):
        if not self.cfg.play:
            selected = self.segment_eligible[env_ids]
            # A reset is a failed segment, including a fall during warmup.
            reset = self._env.episode_length_buf[env_ids] <= 1
            good = (self.segment_age[env_ids] >= 1.5) & (self.segment_samples[env_ids] >= 20)
            good &= self.segment_good[env_ids] / self.segment_samples[env_ids].clamp_min(1) >= .8
            self.speed_gate.observe(int(selected.sum()), int((selected & good & ~reset).sum()))
            # Applies to forward-only AND mixed-axis positive vx samples.
            self.cfg.velocity_ranges.lin_vel_x = (self.cfg.velocity_ranges.lin_vel_x[0], self.speed_gate.cap)
        super()._resample_command(env_ids)
        if self.cfg.play:
            return
        cmd = self._command[env_ids]
        self.segment_eligible[env_ids] = (cmd[:,0] >= .8*self.speed_gate.cap) & (cmd[:,1:3].abs().amax(-1) < 1e-6)
        self.segment_age[env_ids] = 0
        self.segment_samples[env_ids] = 0
        self.segment_good[env_ids] = 0
        pos, yaw = self.read_planar_pose(env_ids)
        self.segment_origin[env_ids] = pos
        self.segment_yaw[env_ids] = yaw

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        if self.cfg.play or env_ids is not None:
            return
        valid = self._env.episode_length_buf > 1
        self.segment_age += valid * self._env.step_dt
        data = self.robot.data
        pos, yaw = self.read_planar_pose(slice(None))
        angle = yaw-self.segment_yaw
        angle = torch.atan2(angle.sin(), angle.cos())
        delta = pos-self.segment_origin
        lateral = -self.segment_yaw.sin()*delta[:,0] + self.segment_yaw.cos()*delta[:,1]
        velocity = data.root_link_lin_vel_b
        good = (velocity[:,0]-self._command[:,0]).abs() <= (.15+.15*self._command[:,0].abs())
        good &= velocity[:,1].abs() <= .15
        good &= data.root_link_ang_vel_b[:,2].abs() <= .2
        good &= angle.abs() <= math.radians(10)
        good &= lateral.abs() <= .15
        measured = valid & self.segment_eligible & (self.segment_age >= 1.)
        self.segment_samples += measured
        self.segment_good += measured & good
        event = self.speed_gate.check(int(self._env.common_step_counter), self.cfg.num_steps_per_iteration)
        if event is not None:
            print('[speed_curriculum] '+json.dumps(event), flush=True)
            if event['promoted']:
                # Discard old-cap segments; new commands enter the new stage on resample.
                self.segment_eligible[:] = False
        self.metrics['speed_cap_mps'].fill_(self.speed_gate.cap)
        self.metrics['gate_success_rate'].fill_(self.speed_gate.last_rate)
        self.metrics['gate_stage'].fill_(self.speed_gate.stage)

    def restore_gate(self, state):
        self.speed_gate = SpeedGate.from_state(state)
        self.cfg.velocity_ranges.lin_vel_x = (self.cfg.velocity_ranges.lin_vel_x[0], self.speed_gate.cap)
        self.segment_eligible[:] = False  # Physical trajectories are not resumed.
