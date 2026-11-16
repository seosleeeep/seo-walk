"""Measure unassisted straight tracking; no waypoint/heading feedback controller."""
import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    parser.add_argument('--vx', type=float, default=1.)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.checkpoint_file.is_file() or not (0 <= args.vx <= 3 and 0 < args.seconds <= 30):
        parser.error('Existing checkpoint, vx in [0,3], seconds in (0,30] required')
    from sjy_vwalk_v2.bootstrap import import_mjlab
    import_mjlab()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from sjy_vwalk_v2.env_cfg import make_env_cfg
    from sjy_vwalk_v2.ppo_cfg import make_ppo_cfg

    cfg = make_env_cfg(True, (args.vx, 0., 0.), distance=None)
    cfg.seed = args.seed
    agent = make_ppo_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    rows = []
    status = 'duration_completed'
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=env.device)
        runner.load(str(args.checkpoint_file), load_cfg={'actor': True}, strict=True, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = wrapper.reset()
        data = env.scene['robot'].data
        def pose():
            x, y = data.root_link_pos_w[0, :2].tolist()
            w, qx, qy, qz = data.root_link_quat_w[0].tolist()
            return x, y, math.atan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        x0, y0, yaw0 = pose()
        c, s = math.cos(yaw0), math.sin(yaw0)
        with torch.inference_mode():
            for step in range(1, math.ceil(args.seconds/env.step_dt)+1):
                obs, _, done, _ = wrapper.step(policy(obs))
                if done.item():
                    status = 'episode_ended'
                    break  # Auto-reset state must not enter this episode's metrics.
                x, y, yaw = pose()
                vx, vy = data.root_link_lin_vel_w[0, :2].tolist()
                dyaw = yaw-yaw0
                rows.append(dict(time_s=step*env.step_dt, command_vx_mps=args.vx,
                    forward_m=c*(x-x0)+s*(y-y0), lateral_m=-s*(x-x0)+c*(y-y0),
                    forward_speed_mps=c*vx+s*vy, lateral_speed_mps=-s*vx+c*vy,
                    heading_deg=math.degrees(math.atan2(math.sin(dyaw),math.cos(dyaw)))))
    finally:
        env.close()
    output = args.checkpoint_file.parent/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    if rows:
        with (output/'velocity.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    steady = [r for r in rows if r['time_s'] >= 2.]
    result = dict(status=status, checkpoint=str(args.checkpoint_file), seed=args.seed,
        command_vx_mps=args.vx, requested_seconds=args.seconds, samples=len(rows),
        last_valid_sample=rows[-1] if rows else None,
        max_abs_lateral_cm=max((abs(r['lateral_m'])*100 for r in rows), default=None),
        max_abs_heading_deg=max((abs(r['heading_deg']) for r in rows), default=None),
        mean_forward_speed_after_2s=sum(r['forward_speed_mps'] for r in steady)/len(steady) if steady else None,
        vx_rmse_after_2s=math.sqrt(sum((r['forward_speed_mps']-args.vx)**2 for r in steady)/len(steady)) if steady else None)
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
