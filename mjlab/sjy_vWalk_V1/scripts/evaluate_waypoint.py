"""Run a single forward waypoint trial and save MuJoCo ground-truth errors."""
import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def errors(x, y, yaw, gx, gy, heading, distance):
    dx, dy = x - gx, y - gy
    c, s = math.cos(heading), math.sin(heading)
    angle = yaw - heading
    return dict(forward_m=distance + c*dx + s*dy, lateral_cm=100*(-s*dx+c*dy),
                goal_error_cm=100*math.hypot(dx, dy),
                heading_deg=math.degrees(math.atan2(math.sin(angle), math.cos(angle))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=2.)
    parser.add_argument("--vx", type=float, default=.5)
    args = parser.parse_args()
    if not args.checkpoint_file.is_file() or not (math.isfinite(args.distance) and args.distance > 0 and 0 < args.vx <= 3):
        parser.error("Check the checkpoint path, positive distance and speed cap (0, 3] m/s")

    from sjy_vwalk_v1.bootstrap import import_mjlab
    import_mjlab()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from sjy_vwalk_v1.env_cfg import make_env_cfg
    from sjy_vwalk_v1.ppo_cfg import make_ppo_cfg

    cfg = make_env_cfg(True, (args.vx, 0., 0.), args.distance)
    cfg.seed = 42
    agent = make_ppo_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    rows = []
    status = "time_limit"
    quiet_time = 0.
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=env.device)
        runner.load(str(args.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        motion = env.command_manager.get_term("motion")
        data = env.scene["robot"].data
        obs, _ = wrapper.reset()
        with torch.inference_mode():
            for step in range(1, math.ceil(cfg.episode_length_s / env.step_dt) + 1):
                obs, _, done, _ = wrapper.step(policy(obs))
                if done.item():
                    status = "episode_ended"
                    break  # The vector wrapper may already contain the next episode's pose.
                if motion.pending[0].item():
                    continue
                gx, gy = motion.goal[0].tolist()
                heading = motion.heading[0].item()
                x, y = data.root_link_pos_w[0, :2].tolist()
                w, qx, qy, qz = data.root_link_quat_w[0].tolist()
                yaw = math.atan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
                speed = data.root_link_vel_w[0, :2].norm().item()
                arrived = motion.arrived[0].item()
                row = dict(time_s=step*env.step_dt, x_m=x, y_m=y, goal_x_m=gx, goal_y_m=gy,
                           speed_mps=speed, stop_command=arrived,
                           **errors(x,y,yaw,gx,gy,heading,args.distance))
                rows.append(row)
                quiet_time = quiet_time + env.step_dt if arrived and speed <= .05 else 0.
                if quiet_time >= 1.:
                    status = "settled_in_tolerance" if row["goal_error_cm"] <= 10 and abs(row["heading_deg"]) <= math.degrees(.09) else "settled_outside_tolerance"
                    break
    finally:
        env.close()
    output = Path("waypoint_results") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    if rows:
        with (output/"trajectory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(status=status, checkpoint=str(args.checkpoint_file.resolve()),
                   distance_m=args.distance, speed_cap_mps=args.vx,
                   max_abs_lateral_cm=max((abs(r["lateral_cm"]) for r in rows), default=None),
                   last_valid_sample=rows[-1] if rows else None)
    report = json.dumps(summary, indent=2)
    (output/"summary.json").write_text(report+"\n", encoding="utf-8")
    print(report, output.resolve(), sep="\n")


if __name__ == "__main__":
    main()
