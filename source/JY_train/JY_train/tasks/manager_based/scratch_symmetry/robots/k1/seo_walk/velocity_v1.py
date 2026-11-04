"""Controlled reward-tuning experiment; preserves v0 and the policy interface.

These coefficients are hypotheses to evaluate, not validated walking gains.
"""

from isaaclab.utils import configclass

from .env_cfg import FlatWoStateEstimationEnvCfg
from .ppo_cfg import SymmetryPPORunnerCfg


@configclass
class VelocityV1EnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # exp(-error_squared / sigma): at vx_cmd=0.2 and vx=0,
        # the x tracking factor changes from exp(-0.04/0.25)=0.852
        # to exp(-0.04/0.05)=0.449. Sigma here is NOT squared again.
        for term in (self.rewards.tracking_lin_vel_x, self.rewards.tracking_lin_vel_y):
            term.params["tracking_sigma"] = 0.05
            term.params["combined_tracking_sigma"] = 0.05

        # Preserve the existing linear tracking weights and momentum term.
        # Swing contact timing alone must not dominate useful translation.
        self.rewards.feet_swing.weight = 1.0  # v0: 3.0
        # Allow larger action changes during early gait learning; other
        # torque, power, joint-velocity and acceleration costs stay active.
        self.rewards.action_rate.weight = -0.75  # v0: -1.5

        # Low-speed forward walking was still strongly penalized for separating
        # the feet fore/aft. Relax alignment, rather than prescribing an
        # arbitrary stride length or increasing vertical foot clearance.
        self.rewards.feet_offset_x.weight = -1.0  # v0: -5.0

        # These regulate trunk height/motion, not head height. Allow more
        # vertical excursion while retaining the existing height target,
        # upright-orientation cost and fall termination.
        self.rewards.base_height.weight = -10.0  # v0: -30.0
        self.rewards.lin_vel_z.weight = -1.0  # v0: -2.0
        self.rewards.root_acc.weight = -5.0e-5  # v0: -1.0e-4

        # More freedom to translate must not reward uncommanded rotation.
        # heading_drift is active only outside commanded turns.
        self.rewards.heading_drift.weight = -2.0  # v0: -1.0
        self.rewards.tracking_ang_vel.weight = 1.5  # v0: 1.0


@configclass
class VelocityV1PlayEnvCfg(VelocityV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.play = True
        self.commands.motion.play_velocity_command = (0.2, 0.0, 0.0)
        self.events.kick_robot = None
        self.events.external_push = None
        self.observations.policy.enable_corruption = False


@configclass
class VelocityV1PPORunnerCfg(SymmetryPPORunnerCfg):
    experiment_name = "seo_walk_v1"
