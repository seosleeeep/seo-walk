"""Gym registrations for K1 locomotion symmetry experiments."""

import gymnasium as gym


def _register(task_id: str, agent_cfg: str, env_cfg: str = "FlatWoStateEstimationEnvCfg"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{agent_cfg}",
        },
    )


_register("seo_walk_v0", "SymmetryPPORunnerCfg")
_register("seo_walk_loss_v0", "MirrorLossPPORunnerCfg")
_register("seo_walk_aug_v0", "MirrorAugPPORunnerCfg")
_register(
    "seo_walk_v0-Play",
    "SymmetryPPORunnerCfg",
    "PlayFlatWoStateEstimationEnvCfg",
)

# Keep the original task reproducible; use separate logs for reward tuning.
for _task_id, _env_class in (
    ("seo_walk_v1", "VelocityV1EnvCfg"),
    ("seo_walk_v1-Play", "VelocityV1PlayEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_v1:{_env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.velocity_v1:VelocityV1PPORunnerCfg",
        },
    )
