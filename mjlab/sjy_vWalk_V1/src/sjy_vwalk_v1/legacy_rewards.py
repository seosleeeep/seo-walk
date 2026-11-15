"""User vv0 reward math, extracted from the original project; data adapted separately."""
from __future__ import annotations
import math
import torch
from .bridge import Selection as SceneEntityCfg
def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi

def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def _roll_pitch_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qw, qx, qy, qz = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    return (roll, pitch)

def _feet_raw_rel(asset: Articulation, foot_body_ids: list[int]) -> torch.Tensor:
    foot_quat_w = asset.data.body_quat_w[:, foot_body_ids]
    base_yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w).unsqueeze(-1)
    foot_yaw = _yaw_from_quat_wxyz(foot_quat_w)
    return _wrap_to_pi(foot_yaw - base_yaw)

def _feet_offset_b(asset: Articulation, foot_body_ids: list[int], feet_distance_ref: float):
    feet_pos = asset.data.body_pos_w[:, foot_body_ids]
    base_yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    dx = feet_pos[:, 0, 0] - feet_pos[:, 1, 0]
    dy = feet_pos[:, 0, 1] - feet_pos[:, 1, 1]
    feet_x_offset = torch.cos(base_yaw) * dx + torch.sin(base_yaw) * dy
    feet_y_offset = -torch.sin(base_yaw) * dx + torch.cos(base_yaw) * dy
    feet_y_offset = feet_y_offset - feet_distance_ref
    return (feet_x_offset, feet_y_offset)

def _staged_tracking_sigma(env, command_name: str, tracking_sigma: float, combined_tracking_sigma: float) -> float:
    """Pure-command 단계와 combined-command 단계의 tracking sigma를 선택."""
    command_term = env.command_manager.get_term(command_name)
    num_steps_per_iteration = max(command_term.cfg.num_steps_per_iteration, 1)
    switch_iteration = command_term.cfg.pure_command_iterations
    iteration = int(env.common_step_counter) // num_steps_per_iteration
    if iteration < switch_iteration:
        return tracking_sigma
    return combined_tracking_sigma

def survival(env):
    return torch.ones(env.num_envs, device=env.device)

def track_lin_vel_x_exp(env, command_name: str, tracking_sigma: float=0.25, combined_tracking_sigma: float=0.1, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    sigma = _staged_tracking_sigma(env, command_name, tracking_sigma, combined_tracking_sigma)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-error / sigma)

def track_lin_vel_y_exp(env, command_name: str, tracking_sigma: float=0.25, combined_tracking_sigma: float=0.1, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    sigma = _staged_tracking_sigma(env, command_name, tracking_sigma, combined_tracking_sigma)
    error = torch.square(command[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-error / sigma)

def track_ang_vel_theta_exp(env, command_name: str, tracking_sigma: float=0.25, combined_tracking_sigma: float=0.1, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    sigma = _staged_tracking_sigma(env, command_name, tracking_sigma, combined_tracking_sigma)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / sigma)

def track_velocity_direction_alignment(env, command_name: str, velocity_scales: tuple[float, float, float]=(1.5, 1.5, 1.6), command_deadband: float=0.05, eps: float=1e-06, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    velocity = torch.stack((asset.data.root_lin_vel_b[:, 0], asset.data.root_lin_vel_b[:, 1], asset.data.root_ang_vel_b[:, 2]), dim=-1)
    scales = command.new_tensor(velocity_scales).clamp_min(eps)
    command_normalized = command / scales
    velocity_normalized = velocity / scales
    dot = torch.sum(command_normalized * velocity_normalized, dim=-1)
    command_norm = torch.linalg.vector_norm(command_normalized, dim=-1)
    velocity_norm = torch.linalg.vector_norm(velocity_normalized, dim=-1)
    cosine = dot / (command_norm * velocity_norm).clamp_min(eps)
    alignment = cosine.clamp(0.0, 1.0)
    active_command = command_norm > command_deadband
    return torch.where(active_command, alignment, torch.zeros_like(alignment))

def base_height_l2(env, target_height: float=0.52, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), sensor_cfg: SceneEntityCfg | None=None):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        base_height = base_height - terrain_height
    return torch.square(base_height - target_height)

def command_conditioned_base_height_l2(env, command_name: str, stand_height: float=0.52, max_height_reduction: float=0.02, velocity_scales: tuple[float, float, float]=(1.5, 1.5, 1.6), yaw_weight: float=0.25, command_deadzone: tuple[float, float, float]=(0.05, 0.05, 0.05), actual_velocity_deadzone: tuple[float, float, float]=(0.1, 0.1, 0.15), asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), sensor_cfg: SceneEntityCfg | None=None):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :3]
    scales = command.new_tensor(velocity_scales).clamp_min(1e-06)
    normalized_command = torch.abs(command) / scales
    command_intensity = torch.sqrt(torch.square(normalized_command[:, 0]) + torch.square(normalized_command[:, 1]) + yaw_weight * torch.square(normalized_command[:, 2])).clamp(0.0, 1.0)
    command_deadzone_tensor = command.new_tensor(command_deadzone)
    command_is_zero = torch.all(torch.abs(command) < command_deadzone_tensor, dim=1)
    command_intensity = torch.where(command_is_zero, torch.zeros_like(command_intensity), command_intensity)
    actual_velocity = torch.stack((asset.data.root_lin_vel_b[:, 0], asset.data.root_lin_vel_b[:, 1], asset.data.root_ang_vel_b[:, 2]), dim=1)
    actual_deadzone_tensor = command.new_tensor(actual_velocity_deadzone)
    filtered_actual_velocity = torch.where(torch.abs(actual_velocity) < actual_deadzone_tensor, torch.zeros_like(actual_velocity), actual_velocity)
    normalized_actual_velocity = torch.abs(filtered_actual_velocity) / scales
    actual_intensity = torch.sqrt(torch.square(normalized_actual_velocity[:, 0]) + torch.square(normalized_actual_velocity[:, 1]) + yaw_weight * torch.square(normalized_actual_velocity[:, 2])).clamp(0.0, 1.0)
    effective_intensity = torch.maximum(command_intensity, actual_intensity)
    height_blend = effective_intensity * effective_intensity * (3.0 - 2.0 * effective_intensity)
    target_height = stand_height - max_height_reduction * height_blend
    base_height = asset.data.root_pos_w[:, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        base_height = base_height - terrain_height
    return torch.square(base_height - target_height)

def projected_gravity_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def joint_torques_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def torque_tiredness(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(asset.data, 'soft_joint_effort_limits'):
        limits = asset.data.soft_joint_effort_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, 'joint_effort_limits'):
        limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    else:
        raise AttributeError('No joint effort limit field found for torque_tiredness.')
    torque_ratio = asset.data.applied_torque[:, asset_cfg.joint_ids] / limits.clamp(min=1e-06)
    return torch.sum(torch.square(torque_ratio).clip(max=1.0), dim=1)

def power_positive(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(power.clip(min=0.0), dim=1)

def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def joint_vel_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def joint_acc_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def root_acc_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    lin_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    ang_acc = asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(lin_acc), dim=(1, 2)) + torch.sum(torch.square(ang_acc), dim=(1, 2))

def action_rate_l2(env):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def _stable_stand_gate(env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float=0.05, lin_velocity_scale: float=0.2, ang_velocity_scale: float=0.5, tilt_scale: float=0.2) -> torch.Tensor:
    """안정적인 정지 상태에서 1, 외란 복구 중에는 0에 가까운 gate.

    명령만 0이라고 무조건 정지 penalty를 적용하면 push 복구를
    방해할 수 있으므로 몸통 속도와 기울기를 함께 확인한다.

    Hard threshold 대신 exponential gate를 사용해서 penalty가
    갑자기 켜지고 꺼지는 것을 방지한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.linalg.vector_norm(command[:, :3], dim=1)
    zero_command = (command_norm < command_threshold).float()
    base_lin_velocity_xy = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    base_ang_velocity = torch.linalg.vector_norm(asset.data.root_ang_vel_b, dim=1)
    roll, pitch = _roll_pitch_from_quat_wxyz(asset.data.root_quat_w)
    base_tilt = torch.sqrt(torch.square(roll) + torch.square(pitch))
    stability_gate = torch.exp(-torch.square(base_lin_velocity_xy / lin_velocity_scale) - torch.square(base_ang_velocity / ang_velocity_scale) - torch.square(base_tilt / tilt_scale))
    return zero_command * stability_gate

def stand_action_rate_l2(env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float=0.05, lin_velocity_scale: float=0.2, ang_velocity_scale: float=0.5, tilt_scale: float=0.2) -> torch.Tensor:
    """안정적인 정지 상태에서 policy action 변경을 억제한다.

    기존 global action_rate와 동일하게 raw policy action 차이를
    사용하지만 정지 상태에서만 추가로 적용한다.
    """
    current_action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    action_difference = current_action - previous_action
    action_rate_cost = torch.sum(torch.square(action_difference), dim=1)
    stand_gate = _stable_stand_gate(env=env, command_name=command_name, asset_cfg=asset_cfg, command_threshold=command_threshold, lin_velocity_scale=lin_velocity_scale, ang_velocity_scale=ang_velocity_scale, tilt_scale=tilt_scale)
    return action_rate_cost * stand_gate

def stand_joint_velocity_l2(env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float=0.05, lin_velocity_scale: float=0.2, ang_velocity_scale: float=0.5, tilt_scale: float=0.2) -> torch.Tensor:
    """안정적인 정지 상태에서 실제 joint velocity를 억제한다."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_velocity_cost = torch.sum(torch.square(joint_velocity), dim=1)
    stand_gate = _stable_stand_gate(env=env, command_name=command_name, asset_cfg=asset_cfg, command_threshold=command_threshold, lin_velocity_scale=lin_velocity_scale, ang_velocity_scale=ang_velocity_scale, tilt_scale=tilt_scale)
    return joint_velocity_cost * stand_gate

def joint_pos_limits_count(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    lower = limits[..., 0]
    upper = limits[..., 1]
    return torch.sum(((joint_pos < lower) | (joint_pos > upper)).float(), dim=1)

def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def feet_swing(env, command_name: str, sensor_cfg: SceneEntityCfg, swing_period: float=0.2, threshold: float=1.0):
    command_term = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    phase = command_term.gait_process
    freq = command_term.gait_frequency
    left_swing = (torch.abs(phase - 0.25) < 0.5 * swing_period) & (freq > 1e-08)
    right_swing = (torch.abs(phase - 0.75) < 0.5 * swing_period) & (freq > 1e-08)
    return (left_swing & ~feet_contact[:, 0]).float() + (right_swing & ~feet_contact[:, 1]).float()

def foot_height_trajectory_exp(env, command_name: str, asset_cfg: SceneEntityCfg, maximum_height: float=0.06, kappa: float=4.0, tracking_sigma: float=0.0004, command_threshold: float=0.05):
    """Track the requested periodic left/right swing-foot height trajectory."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    desired_height = _von_mises_foot_height_profiles(command_term.gait_process, maximum_height, kappa)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    actual_height = foot_z - torch.min(foot_z, dim=1, keepdim=True).values
    error = torch.mean(torch.square(actual_height - desired_height), dim=1)
    tracking = torch.exp(-error / tracking_sigma)
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
    return tracking * moving.float()

def raibert_foot_placement_exp(env, command_name: str, foot_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg, stance_fraction: float=0.5, velocity_error_gain: float=0.2, tracking_sigma: float=0.04, kappa: float=4.0, command_threshold: float=0.05):
    """Track swing-foot XY targets computed with the Raibert heuristic.

    p_target = p_hip + T_stance/2 * v_current
                         + K * (v_current - v_desired)
    """
    asset: Articulation = env.scene[foot_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    phase = command_term.gait_process
    foot_pos_xy_w = asset.data.body_pos_w[:, foot_cfg.body_ids, :2]
    hip_pos_xy_w = asset.data.body_pos_w[:, hip_cfg.body_ids, :2]
    if foot_pos_xy_w.shape[1] != 2 or hip_pos_xy_w.shape[1] != 2:
        raise ValueError('Raibert foot placement requires exactly two feet and two hips.')
    current_velocity_w = asset.data.root_lin_vel_w[:, :2]
    base_yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    cos_yaw = torch.cos(base_yaw)
    sin_yaw = torch.sin(base_yaw)
    desired_velocity_w = torch.stack((cos_yaw * command[:, 0] - sin_yaw * command[:, 1], sin_yaw * command[:, 0] + cos_yaw * command[:, 1]), dim=1)
    frequency = command_term.gait_frequency.clamp_min(1e-06)
    stance_time = stance_fraction / frequency
    placement_offset = 0.5 * stance_time.unsqueeze(1) * current_velocity_w + velocity_error_gain * (current_velocity_w - desired_velocity_w)
    target_xy_w = hip_pos_xy_w + placement_offset.unsqueeze(1)
    error_per_foot = torch.sum(torch.square(foot_pos_xy_w - target_xy_w), dim=-1)
    tracking_per_foot = torch.exp(-error_per_foot / tracking_sigma)
    swing_weights = _von_mises_foot_height_profiles(phase, maximum_height=1.0, kappa=kappa)
    weighted_tracking = torch.sum(tracking_per_foot * swing_weights, dim=1) / swing_weights.sum(dim=1).clamp_min(1e-06)
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
    return weighted_tracking * moving.float()

def _compact_von_mises_foot_height_profiles(phase: torch.Tensor, maximum_height: float, kappa: float) -> torch.Tensor:
    """Tail을 제거한 von-Mises 기반 좌우 발 높이 profile.

    기존 von-Mises의 다음 특성은 유지한다.

    - Left swing center: phase=0.25
    - Right swing center: phase=0.75
    - maximum height
    - kappa 기반 종 모양
    - 좌우 0.5 phase 차이

    stance 구간의 양수 tail만 제거하고, swing 경계에서 높이,
    속도, 가속도가 0이 되도록 quintic smootherstep을 적용한다.

    출력 shape:
        (num_envs, 2)

    출력 순서:
        [:, 0] = left foot target
        [:, 1] = right foot target
    """
    if maximum_height < 0.0:
        raise ValueError(f'maximum_height must be non-negative, got {maximum_height}.')
    if kappa <= 0.0:
        raise ValueError(f'kappa must be positive, got {kappa}.')
    centers = phase.new_tensor([0.25, 0.75])
    phase_per_foot = phase.unsqueeze(1) - centers.unsqueeze(0)
    raw_profile = torch.exp(kappa * (torch.cos(2.0 * torch.pi * phase_per_foot) - 1.0))
    boundary_value = math.exp(-kappa)
    normalized_profile = torch.clamp((raw_profile - boundary_value) / (1.0 - boundary_value), min=0.0, max=1.0)
    smooth_profile = normalized_profile.pow(3) * (10.0 - 15.0 * normalized_profile + 6.0 * normalized_profile.square())
    target_height = maximum_height * smooth_profile
    return target_height

def footswing_trajectory_tracking_l2(env, command_name: str, foot_cfg: SceneEntityCfg, maximum_height: float=0.13, kappa: float=1.0, height_error_scale: float=5.0, command_threshold: float=0.05) -> torch.Tensor:
    """좌우 발의 높이 trajectory만 추종하는 L2 cost.

    XY/Raibert error는 계산하지 않는다.

    목표 trajectory:
        phase=0.00:
            left=0, right=0

        phase=0.25:
            left=maximum_height, right=0

        phase=0.50:
            left=0, right=0

        phase=0.75:
            left=0, right=maximum_height

    반환값은 양수 cost이므로 RewTerm weight는 음수여야 한다.
    """
    asset: Articulation = env.scene[foot_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    foot_z = asset.data.body_pos_w[:, foot_cfg.body_ids, 2]
    if foot_z.shape[1] != 2:
        raise ValueError(f'footswing_trajectory_tracking_l2 requires exactly two feet, but got shape {foot_z.shape}.')
    target_height = _compact_von_mises_foot_height_profiles(phase=command_term.gait_process, maximum_height=maximum_height, kappa=kappa)
    lower_foot_z = torch.min(foot_z, dim=1, keepdim=True).values
    actual_height = foot_z - lower_foot_z
    height_error_per_foot = torch.square(actual_height - target_height)
    height_error = torch.sum(height_error_per_foot, dim=1)
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
    trajectory_cost = height_error_scale * height_error * moving.float()
    return trajectory_cost

def feet_slip(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float=1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.square(foot_vel_xy), dim=-1)
    return torch.sum(slip * feet_contact.float(), dim=1)

def feet_yaw_l2(env, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    return torch.sum(torch.square(feet_yaw_rel), dim=1)

def feet_roll_l2(env, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _ = _roll_pitch_from_quat_wxyz(asset.data.body_quat_w[:, asset_cfg.body_ids])
    return torch.sum(torch.square(_wrap_to_pi(roll)), dim=1)

def feet_distance(env, asset_cfg: SceneEntityCfg, feet_distance_ref: float=0.18):
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    feet_separation = torch.linalg.norm(feet_pos_xy[:, 0] - feet_pos_xy[:, 1], dim=1)
    return torch.clamp(feet_distance_ref - feet_separation, min=0.0)

def stand_feet_flat_l2(env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float=0.05, speed_threshold: float=0.2):
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_is_zero = torch.linalg.vector_norm(command[:, :3], dim=1) <= command_threshold
    actual_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    nearly_stopped = actual_speed <= speed_threshold
    activate = command_is_zero & nearly_stopped
    roll, pitch = _roll_pitch_from_quat_wxyz(asset.data.body_quat_w[:, asset_cfg.body_ids])
    error = torch.sum(torch.square(_wrap_to_pi(roll)) + torch.square(_wrap_to_pi(pitch)), dim=1)
    return error * activate.float()

def root_feet_lateral_diff(env, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, contact_threshold: float=1.0, base_deadzone: float=0.02, vy_deadzone_gain: float=0.04, yaw_deadzone_gain: float=0.03, max_vy: float=1.5, max_yaw: float=1.6):
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_center_w = torch.mean(asset.data.body_pos_w[:, asset_cfg.body_ids, :2], dim=1)
    root_to_feet_w = asset.data.root_pos_w[:, :2] - feet_center_w
    yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    lateral_offset = -torch.sin(yaw) * root_to_feet_w[:, 0] + torch.cos(yaw) * root_to_feet_w[:, 1]
    vy_ratio = torch.clamp(torch.abs(command[:, 1]) / max_vy, max=1.0)
    yaw_ratio = torch.clamp(torch.abs(command[:, 2]) / max_yaw, max=1.0)
    allowed_offset = base_deadzone + vy_deadzone_gain * vy_ratio + yaw_deadzone_gain * yaw_ratio
    error = torch.clamp(torch.abs(lateral_offset) - allowed_offset, min=0.0)
    forces = sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    double_support = torch.all(feet_contact, dim=1)
    return torch.square(error) * double_support.float()

def joint_roll_yaw_symmetry_l2(env, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    pairs = [('Left_Hip_Roll', 'Right_Hip_Roll'), ('Left_Hip_Yaw', 'Right_Hip_Yaw'), ('Left_Ankle_Roll', 'Right_Ankle_Roll')]
    penalty = torch.zeros(env.num_envs, device=env.device)
    for left_name, right_name in pairs:
        left_id = asset.joint_names.index(left_name)
        right_id = asset.joint_names.index(right_name)
        q_left = joint_pos[:, left_id]
        q_right = joint_pos[:, right_id]
        penalty += torch.square(q_left + q_right)
    return penalty

def feet_offset_x_l1(env, command_name: str, asset_cfg: SceneEntityCfg, max_forward_vel: float=1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_x_offset, _ = _feet_offset_b(asset, asset_cfg.body_ids, feet_distance_ref=0.18)
    error = feet_x_offset - command[:, 8]
    penalty = torch.clip(torch.abs(error), min=0.0, max=0.1)
    vel_scale = torch.clamp((1.0 - torch.abs(command[:, 0]) / max_forward_vel) ** 2, min=0.0, max=1.0)
    return penalty * vel_scale

def feet_offset_y_l1(env, command_name: str, asset_cfg: SceneEntityCfg, feet_distance_ref: float=0.18, max_lateral_vel: float=1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, feet_y_offset = _feet_offset_b(asset, asset_cfg.body_ids, feet_distance_ref)
    error = feet_y_offset - command[:, 9]
    penalty = torch.clip(torch.abs(error), min=0.0, max=0.1)
    vel_scale = torch.clamp((1.0 - torch.abs(command[:, 1]) / max_lateral_vel) ** 2, min=0.0, max=1.0)
    return penalty * vel_scale

def foot_yaw_l_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    error = _wrap_to_pi(feet_yaw_rel[:, 0] - command[:, 4])
    return torch.square(error)

def foot_yaw_r_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    error = _wrap_to_pi(feet_yaw_rel[:, 1] - command[:, 5])
    return torch.square(error)

def heading_drift_l2(env, command_name: str, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), deadband: float=0.1) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    if not hasattr(env, 'ref_yaw') or env.ref_yaw.shape != yaw.shape:
        env.ref_yaw = yaw.clone()
    turning = torch.abs(command[:, 2]) >= deadband
    reset = env.episode_length_buf <= 1
    update_ref = reset | turning
    env.ref_yaw[update_ref] = yaw[update_ref]
    yaw_error = _wrap_to_pi(yaw - env.ref_yaw)
    no_turn = ~turning
    return torch.square(yaw_error) * no_turn.float()

def feet_pitch_l2(env, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch = _roll_pitch_from_quat_wxyz(asset.data.body_quat_w[:, asset_cfg.body_ids])
    return torch.sum(torch.square(_wrap_to_pi(pitch)), dim=1)

def feet_yaw_diff_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    commanded_diff = command[:, 5] - command[:, 4]
    actual_diff = feet_yaw_rel[:, 1] - feet_yaw_rel[:, 0]
    return torch.square(_wrap_to_pi(actual_diff - commanded_diff))

def feet_yaw_mean_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    commanded_mean = 0.5 * (command[:, 5] + command[:, 4])
    actual_mean = torch.mean(feet_yaw_rel, dim=1)
    return torch.square(_wrap_to_pi(actual_mean - commanded_mean))

def pure_x_lateral_drift_l2(env, command_name: str, command_deadband: float=0.05, zero_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = (torch.abs(command[:, 0]) > command_deadband) & (torch.abs(command[:, 1]) < zero_deadband) & (torch.abs(command[:, 2]) < zero_deadband)
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * mask.float()

def pure_x_yaw_drift_l2(env, command_name: str, command_deadband: float=0.05, zero_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = (torch.abs(command[:, 0]) > command_deadband) & (torch.abs(command[:, 1]) < zero_deadband) & (torch.abs(command[:, 2]) < zero_deadband)
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

def pure_y_forward_drift_l2(env, command_name: str, command_deadband: float=0.05, zero_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = (torch.abs(command[:, 1]) > command_deadband) & (torch.abs(command[:, 0]) < zero_deadband) & (torch.abs(command[:, 2]) < zero_deadband)
    return torch.square(asset.data.root_lin_vel_b[:, 0]) * mask.float()

def pure_y_yaw_drift_l2(env, command_name: str, command_deadband: float=0.05, zero_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = (torch.abs(command[:, 1]) > command_deadband) & (torch.abs(command[:, 0]) < zero_deadband) & (torch.abs(command[:, 2]) < zero_deadband)
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

def zero_yaw_command_ang_vel_l2(env, command_name: str, zero_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = torch.abs(command[:, 2]) < zero_deadband
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

def xy_perpendicular_velocity_l2(env, command_name: str, command_deadband: float=0.05, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_xy = command[:, :2]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_norm = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
    active = cmd_norm.squeeze(-1) > command_deadband
    cmd_dir = cmd_xy / cmd_norm.clamp(min=1e-06)
    vel_parallel = torch.sum(vel_xy * cmd_dir, dim=1, keepdim=True) * cmd_dir
    vel_perp = vel_xy - vel_parallel
    return torch.sum(torch.square(vel_perp), dim=1) * active.float()
