"""Read-only V1 policy diagnostics. No training or policy/robot configuration edits."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(states, joints, warmup):
    steady = [r for r in states if r['time_s'] >= warmup]
    result = {'valid_samples': len(states), 'steady_samples': len(steady),
              'last_valid_sample': states[-1] if states else None}
    for field in ('body_vx_mps', 'body_vy_mps', 'body_wz_radps',
                  'world_wz_radps', 'forward_speed_mps', 'planar_speed_mps'):
        result['mean_' + field] = mean([r[field] for r in steady])
    result['body_vx_rmse_mps'] = (
        math.sqrt(mean([(r['body_vx_mps'] - r['command_vx_mps'])**2 for r in steady]))
        if steady else None)
    result['mean_abs_body_wz_radps'] = mean([abs(r['body_wz_radps']) for r in steady])
    result['max_abs_heading_deg'] = max((abs(r['heading_deg']) for r in states), default=None)
    result['max_abs_lateral_cm'] = max((100*abs(r['lateral_m']) for r in states), default=None)
    per_joint = {}
    for name in sorted({r['joint'] for r in joints}):
        rows = [r for r in joints if r['joint'] == name and r['time_s'] >= warmup]
        if not rows:
            continue
        q = [r['q_rad'] for r in rows]
        per_joint[name] = dict(mean_q_rad=mean(q), min_q_rad=min(q), max_q_rad=max(q),
            range_q_rad=max(q)-min(q), mean_default_offset_rad=mean([r['q_minus_default_rad'] for r in rows]),
            rms_tau_nm=math.sqrt(mean([r['tau_nm']**2 for r in rows])),
            max_abs_tau_nm=max(abs(r['tau_nm']) for r in rows),
            cap_nm=rows[0]['cap_nm'], near_cap_fraction=mean([r['near_cap'] for r in rows]),
            max_abs_tau_over_cap=max(r['abs_tau_over_cap'] for r in rows))
    result['joints'] = per_joint
    pairs = {}
    for name, left in per_joint.items():
        if 'Left_' not in name:
            continue
        right_name = name.replace('Left_', 'Right_')
        if right_name not in per_joint:
            continue
        right = per_joint[right_name]
        sign = -1 if '_Roll' in name or '_Yaw' in name else 1
        pairs[name] = dict(right=right_name, mirror_sign=sign,
            mean_pose_mirror_bias_rad=left['mean_q_rad']-sign*right['mean_q_rad'],
            range_difference_rad=left['range_q_rad']-right['range_q_rad'])
    result['left_right_mean_pose_comparison'] = pairs
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', type=Path, required=True)
    parser.add_argument('--vx', type=float, default=1.)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--warmup', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.checkpoint_file.is_file() or not (0 <= args.vx <= 3 and 0 <= args.warmup < args.seconds <= 30):
        parser.error('Existing model, vx [0,3], 0 <= warmup < seconds <= 30 required')
    from sjy_vwalk_v1.bootstrap import import_mjlab
    import_mjlab()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from sjy_vwalk_v1.env_cfg import make_env_cfg
    from sjy_vwalk_v1.ppo_cfg import make_ppo_cfg
    from sjy_vwalk_v1.robot import POLICY_JOINT_NAMES, effort_limit

    checkpoint = args.checkpoint_file.resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    parent = args.output_dir or checkpoint.parent/'diagnostics'
    out = parent/f'{checkpoint.stem}_vx{args.vx:g}_seed{args.seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    cfg = make_env_cfg(True, (args.vx, 0., 0.), distance=None)
    cfg.seed = args.seed
    agent = make_ppo_cfg()
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        command_vx_mps=args.vx, requested_seconds=args.seconds, warmup_seconds=args.warmup,
        seed=args.seed, device=args.device, source_sha256={
            str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'src').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.xml')},
        script_sha256=sha256(__file__),
        torque_note='50Hz snapshots of qfrc_actuator / nominal V1 constant torque cap; near_cap >= 0.95. Not a 500Hz saturation duty cycle.',
        symmetry_note='Mean pose and range diagnostics only; not instantaneous or phase-aligned gait symmetry.',
        status='initializing')
    states, joints = [], []
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=env.device)
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = wrapper.reset()
        robot = env.scene['robot']
        data = robot.data
        ids, names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
        if list(names) != list(POLICY_JOINT_NAMES):
            raise RuntimeError('Joint order mismatch')
        if env.command_manager.get_term('motion').waypoint.any():
            raise RuntimeError('Waypoint controller must be disabled')
        caps = [effort_limit(n) for n in names]
        def pose():
            x, y = data.root_link_pos_w[0, :2].tolist()
            w, qx, qy, qz = data.root_link_quat_w[0].tolist()
            return x, y, math.atan2(2*(w*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        x0, y0, yaw0 = pose()
        c, s = math.cos(yaw0), math.sin(yaw0)
        metadata.update(status='duration_completed', policy_dt_s=env.step_dt,
                        joint_names=list(names), termination_terms=[])
        with torch.inference_mode():
            for step in range(1, math.ceil(args.seconds/env.step_dt)+1):
                action = policy(obs)
                obs, _, done, _ = wrapper.step(action)
                if done.item():
                    metadata['status'] = 'episode_ended'
                    metadata['termination_step'] = step
                    metadata['termination_terms'] = [n for n in env.termination_manager.active_terms
                        if env.termination_manager.get_term(n)[0].item()]
                    break  # Auto-reset may have overwritten the robot state.
                t = step*env.step_dt
                x, y, yaw = pose()
                vx, vy = data.root_link_lin_vel_w[0, :2].tolist()
                bx, by, bz = data.root_link_lin_vel_b[0].tolist()
                wx, wy, wz = data.root_link_ang_vel_b[0].tolist()
                row = dict(time_s=t, command_vx_mps=args.vx,
                    forward_m=c*(x-x0)+s*(y-y0), lateral_m=-s*(x-x0)+c*(y-y0),
                    forward_speed_mps=c*vx+s*vy, lateral_speed_mps=-s*vx+c*vy,
                    planar_speed_mps=math.hypot(vx,vy), body_vx_mps=bx, body_vy_mps=by,
                    body_vz_mps=bz, body_wx_radps=wx, body_wy_radps=wy, body_wz_radps=wz,
                    world_wz_radps=data.root_link_ang_vel_w[0,2].item(),
                    heading_deg=math.degrees(math.atan2(math.sin(yaw-yaw0),math.cos(yaw-yaw0))),
                    root_height_m=(data.root_link_pos_w[0,2]-env.scene.env_origins[0,2]).item())
                if not all(math.isfinite(v) for v in row.values()):
                    raise RuntimeError('Nonfinite root diagnostic')
                q = data.joint_pos[0,ids].tolist()
                qd = data.joint_vel[0,ids].tolist()
                tau = data.qfrc_actuator[0,ids].tolist()
                defaults = data.default_joint_pos[0,ids].tolist()
                targets = data.joint_pos_target[0,ids].tolist()
                if not all(math.isfinite(v) for values in (q,qd,tau,defaults,targets) for v in values):
                    raise RuntimeError('Nonfinite joint diagnostic')
                states.append(row)
                for i, name in enumerate(names):
                    ratio = abs(tau[i])/caps[i]
                    joints.append(dict(time_s=t, joint=name, q_rad=q[i], qd_radps=qd[i],
                        default_q_rad=defaults[i], q_minus_default_rad=q[i]-defaults[i],
                        target_q_rad=targets[i], tau_nm=tau[i], cap_nm=caps[i],
                        abs_tau_over_cap=ratio, near_cap=int(ratio >= .95)))
    except BaseException as exc:
        metadata.update(status='error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_csv(out/'state.csv', states)
        write_csv(out/'joints.csv', joints)
        metadata['metrics'] = summarize(states,joints,args.warmup)
        (out/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
        print('Saved diagnostics:', out, flush=True)
        if env is not None:
            env.close()
    print(json.dumps({k:v for k,v in metadata['metrics'].items()
        if k not in ('joints','left_right_mean_pose_comparison')}, indent=2))


if __name__ == '__main__':
    main()
