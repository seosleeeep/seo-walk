"""Measure arm swing in the trunk frame during a fixed-command rollout."""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--vx", type=float, default=1.0)
    args = parser.parse_args()
    if not args.checkpoint_file.is_file():
        parser.error("Checkpoint file does not exist")
    from sjy_vwalk_v1.bootstrap import import_mjlab
    import_mjlab()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from sjy_vwalk_v1.env_cfg import make_env_cfg
    from sjy_vwalk_v1.ppo_cfg import make_ppo_cfg

    cfg = make_env_cfg(True, (args.vx, 0., 0.))
    cfg.scene.num_envs = 1
    cfg.seed = 42
    agent = make_ppo_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=env.device)
        runner.load(str(args.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]
        ids = [robot.body_names.index(n) for n in
               ("left_hand_link", "right_hand_link", "Left_Arm_3", "Right_Arm_3")]
        obs, _ = wrapper.reset()
        warmup = round(2 / env.step_dt)
        samples = []
        with torch.inference_mode():
            for step in range(warmup + round(10 / env.step_dt)):
                obs, _, done, _ = wrapper.step(policy(obs))
                if done.item():
                    raise RuntimeError("Episode ended before completing the measurement")
                if step < warmup:
                    continue
                data = robot.data
                w, x, y, z = data.root_link_quat_w[0].unbind()
                forward = torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)))
                offset = data.body_link_pos_w[0, ids] - data.root_link_pos_w[0]
                samples.append((offset @ forward) * 100)
        pos = torch.stack(samples)
        for i, label in ((0, "hand_link_origin"), (2, "elbow_pitch_origin")):
            left, right = pos[:, i:i+2].mean(0).tolist()
            span_l, span_r = (pos[:, i:i+2].amax(0) - pos[:, i:i+2].amin(0)).tolist()
            print(f"{label}: left_mean={left:+.2f} right_mean={right:+.2f} "
                  f"left_minus_right={left-right:+.2f} cm | swing_span_L/R={span_l:.2f}/{span_r:.2f} cm")
    finally:
        env.close()


if __name__ == "__main__":
    main()
