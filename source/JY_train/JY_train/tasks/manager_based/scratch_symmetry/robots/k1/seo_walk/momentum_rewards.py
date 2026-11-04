"""Seo-only centroidal momentum cost (Isaac Lab 2.x / PhysX).

This is a normalized quadratic experiment, not the paper's clipped exponential.
It does not change action dimensions or enable arm actuators.
"""
from __future__ import annotations

import math
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def centroidal_momentum_w(mass, inertia_b, quat_w, pos_w, lin_vel_w, ang_vel_w):
    """Whole-body angular momentum about its CoM, in world coordinates.

    inertia_b is about each link's CoM, expressed in its ACTOR frame
    (PhysX get_inertias); quat_w must therefore be the LINK quaternion,
    not the principal-inertia quaternion. All positions/velocities are CoM data.
    Returns (num_envs, 3), in kg m^2/s, including spin AND orbital momentum.
    """
    mass = mass.unsqueeze(-1)
    total_mass = mass.sum(dim=1, keepdim=True)
    com = (mass * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_vel = (mass * lin_vel_w).sum(dim=1, keepdim=True) / total_mass
    omega_b = quat_apply_inverse(quat_w, ang_vel_w)
    spin_b = torch.matmul(inertia_b, omega_b.unsqueeze(-1)).squeeze(-1)
    spin_w = quat_apply(quat_w, spin_b)
    orbital_w = torch.cross(pos_w - com, mass * (lin_vel_w - com_vel), dim=-1)
    return (spin_w + orbital_w).sum(dim=1)


class CentroidalYawMomentumPenalty(ManagerTermBase):
    """Positive cost; configure a NEGATIVE reward weight.

    Read randomized physical properties lazily, after startup/reset events.
    Reset invalidates the cache. If interval events change mass/inertia,
    invalidate it there too. Current seo_walk changes these only at startup.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        for key in ("momentum_scale", "yaw_command_threshold"):
            value = cfg.params[key]
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be finite and positive")
        self._mass = None
        self._inertia = None

    def reset(self, env_ids=None):
        self._mass = None
        self._inertia = None

    def __call__(
        self, env, command_name: str, momentum_scale: float,
        yaw_command_threshold: float, asset_cfg: SceneEntityCfg,
    ):
        # Always include every rigid body, irrespective of controlled joints.
        if self._mass is None:
            self._mass = self.asset.root_physx_view.get_masses().to(env.device).clone()
            self._inertia = self.asset.root_physx_view.get_inertias().to(env.device).clone()
            if not torch.isfinite(self._mass).all() or (self._mass < 0).any():
                raise ValueError("Invalid robot body masses")
            if (self._mass.sum(dim=1) <= 0).any():
                raise ValueError("Robot total mass must be positive")
            if not torch.isfinite(self._inertia).all():
                raise ValueError("Invalid robot body inertias")
            self._inertia = self._inertia.reshape(*self._mass.shape, 3, 3)

        data = self.asset.data
        momentum = centroidal_momentum_w(
            self._mass, self._inertia, data.body_quat_w,
            data.body_com_pos_w, data.body_com_lin_vel_w, data.body_ang_vel_w,
        )
        yaw_command = env.command_manager.get_command(command_name)[:, 2]
        # Zero at/above the turn threshold; smoothly fade near it.
        gate = (1.0 - (yaw_command / yaw_command_threshold).square()).clamp(0.0, 1.0)
        return gate * (momentum[:, 2] / momentum_scale).square()
