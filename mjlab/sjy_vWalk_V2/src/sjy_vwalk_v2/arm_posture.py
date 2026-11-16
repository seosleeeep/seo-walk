"""Continuous command-conditioned arm cost, ordered as K1_ARM_JOINT_NAMES.

Only the reward regards activity <= 0.1 as standing. The command, phase,
waypoint controller and action/observation contract are NOT changed.
"""
import torch


def arm_posture_std(command, stand_threshold=0.1, walk_knot=1.0,
                    run_knot=2.0, stand_std=0.05, walk_std=0.15,
                    run_std=(0.5, 0.2, 0.35, 0.35) * 2):
    """Return [batch, 8] standard deviations (radians), not joint limits.

    activity = norm([vx, vy]) + abs(yaw_rate), a heuristic activity score.
    The four per-side entries are Shoulder Pitch/Roll, Elbow Pitch/Yaw.
    """
    if not (0 <= stand_threshold < walk_knot < run_knot):
        raise ValueError('Expected 0 <= stand_threshold < walk_knot < run_knot')
    if len(run_std) != 8 or any(x <= 0 for x in run_std):
        raise ValueError('run_std must contain eight positive values')
    if run_std[:4] != run_std[4:]:
        raise ValueError('Left/right corresponding arm tolerances must match')
    if not (0 < stand_std <= walk_std) or any(x < walk_std for x in run_std):
        raise ValueError('Tolerances must be positive and nondecreasing')
    activity = torch.linalg.vector_norm(command[..., :2], dim=-1) + command[..., 2].abs()
    walk_mix = ((activity - stand_threshold) / (walk_knot - stand_threshold)).clamp(0., 1.)
    run_mix = ((activity - walk_knot) / (run_knot - walk_knot)).clamp(0., 1.)
    base = stand_std + (walk_std - stand_std) * walk_mix
    run = command.new_tensor(run_std)
    return base[..., None] + run_mix[..., None] * (run - walk_std)


def arm_posture_cost(joint_pos, default_joint_pos, command, **kwargs):
    """Unweighted nonnegative mean cost; reward config supplies weight -0.1.

    Use the environment-specific randomized default, the same reference as
    observations and action offsets. No clipping or forced arm trajectory.
    """
    if joint_pos.shape[-1] != 8 or default_joint_pos.shape[-1] != 8:
        raise ValueError('Expected eight arm joints in K1_ARM_JOINT_NAMES order')
    std = arm_posture_std(command, **kwargs)
    return ((joint_pos - default_joint_pos) / std).square().mean(-1)
