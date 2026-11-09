"""20-action observations, reward dispatch, reset and disturbance state."""
from pathlib import Path
import json
import math
import torch
from . import legacy_rewards
from .bridge import bridge,resolve
from .joint_config import K1_ARM_JOINT_NAMES,K1_ARM_DEFAULT_POS
from .robot import POLICY_JOINT_NAMES

REWARDS=json.loads((Path(__file__).parent/'reward_config.json').read_text())

def joint_ids(env):
    if not hasattr(env,'_seo_policy_joint_ids'):
        ids,names=env.scene['robot'].find_joints(POLICY_JOINT_NAMES,preserve_order=True)
        assert names==POLICY_JOINT_NAMES
        env._seo_policy_joint_ids=ids
    return env._seo_policy_joint_ids

def arm_ids(env):
    if not hasattr(env,'_seo_arm_ids'):
        env._seo_arm_ids=env.scene['robot'].find_joints(K1_ARM_JOINT_NAMES,preserve_order=True)[0]
    return env._seo_arm_ids

def reward(env,term_name):
    entry=REWARDS[term_name]
    if term_name=='arm_torque':
        return env.scene['robot'].data.qfrc_actuator[:,arm_ids(env)].square().sum(-1)
    if term_name=='arm_deviation':
        q=env.scene['robot'].data.joint_pos[:,arm_ids(env)]
        ref=q.new_tensor([K1_ARM_DEFAULT_POS[n] for n in K1_ARM_JOINT_NAMES])
        weights = q.new_tensor([1.0, 16.0, 4.0, 4.0] * 2)
        return ((q - ref).square() * weights).mean(-1)
    if not hasattr(env,'_seo_reward_params'):
        env._seo_reward_params={name:{k:resolve(v,env) for k,v in item.get('params',{}).items()}
            for name,item in REWARDS.items()}
    return getattr(legacy_rewards,entry['func'])(bridge(env),**env._seo_reward_params[term_name])

def command(env):
    return env.command_manager.get_command('motion')

def straight_pose_mirror(env):
    """Mean squared joint error against the mirrored pose half a cycle ago."""
    from .symmetry import _joint_permutation, _mirror_joints

    cmd = command(env)
    q = env.scene['robot'].data.joint_pos[:, joint_ids(env)]
    # Walking frequency is fixed by this task's command configuration.
    frequency = env.command_manager.get_term('motion').cfg.default_internal_parameters[0]
    lag = max(1, round(0.5 / (frequency * env.step_dt)))
    if not hasattr(env, '_seo_mirror_history'):
        env._seo_mirror_history = q.new_zeros(lag, *q.shape)
        env._seo_mirror_age = torch.zeros_like(cmd[:, 0], dtype=torch.long)
        env._seo_mirror_command = cmd[:, :4].clone()
        env._seo_mirror_map = _joint_permutation(env, q.device)
    slot = int(env.common_step_counter) % lag
    past = env._seo_mirror_history[slot]
    straight = ((cmd[:, 0] > 0.05) & (cmd[:, 1:3].abs().amax(-1) < 0.05)
                & ((cmd[:, 3] - frequency).abs() < 1e-5))
    same_command = (cmd[:, :4] == env._seo_mirror_command).all(-1)
    valid = straight & same_command & (env._seo_mirror_age >= lag)
    target = _mirror_joints(past, *env._seo_mirror_map)
    cost = (q - target).square().mean(-1)
    past.copy_(q.detach())
    env._seo_mirror_age = torch.where(straight, torch.where(
        same_command, env._seo_mirror_age + 1, 1), 0).clamp_max(lag)
    env._seo_mirror_command.copy_(cmd[:, :4])
    return torch.where(valid, cost, 0.)

def gait_phase(env):
    phase=env.command_manager.get_term('motion').gait_process*2*math.pi
    return torch.stack([torch.cos(phase),torch.sin(phase)],-1)

def gravity(env):
    return env.scene['robot'].data.projected_gravity_b

def angular_velocity(env):
    return env.scene['robot'].data.root_link_ang_vel_b

def joint_position(env):
    data=env.scene['robot'].data
    return (data.joint_pos-data.default_joint_pos)[:,joint_ids(env)]

def joint_velocity(env):
    data=env.scene['robot'].data
    return (data.joint_vel-data.default_joint_vel)[:,joint_ids(env)]

def actions(env):
    return env.action_manager.action

def privileged(env):
    robot=env.scene['robot']
    data=robot.data
    ids=robot.indexing.body_ids
    trunk=robot.find_bodies('Trunk')[0][0]
    index=ids[trunk]
    # Nominal properties come from the shared MJCF, not a different task.
    model=env.sim.mj_model
    ipos=env.sim.model.body_ipos[:,index]
    mass=env.sim.model.body_mass[:,index]
    ipos=ipos.expand(env.num_envs,-1)
    mass=mass.expand(env.num_envs)
    ref=torch.as_tensor(model.body_ipos[int(index)],device=env.device,dtype=ipos.dtype)
    mass_com=torch.cat([ipos-ref,(mass/float(model.body_mass[int(index)]))[:,None]],-1)
    force,torque=push_state(env)
    return torch.cat([mass_com,data.root_link_lin_vel_b,data.root_link_pos_w[:,2:3],force,torque],-1)

def push_state(env):
    if not hasattr(env,'_seo_force'):
        env._seo_force=torch.zeros(env.num_envs,3,device=env.device)
        env._seo_torque=torch.zeros_like(env._seo_force)
        env._seo_next_push=torch.full((env.num_envs,),300,device=env.device,dtype=torch.long)
        env._seo_end_push=torch.zeros_like(env._seo_next_push)
    return env._seo_force,env._seo_torque

def disturbance(env,env_ids):
    force,torque=push_state(env)
    step=int(env.common_step_counter)
    start=step>=env._seo_next_push
    count=int(start.sum())
    if count:
        force[start]=torch.randn(count,3,device=env.device)*6.
        torque[start]=torch.randn(count,3,device=env.device)*.6
        env._seo_next_push[start]=step+300
        env._seo_end_push[start]=step+50
    end=step>=env._seo_end_push
    force[end]=0.
    torque[end]=0.
    robot=env.scene['robot']
    robot.write_external_wrench_to_sim(force[:,None],torque[:,None],
        body_ids=robot.find_bodies('Trunk')[0])

def reset_port_state(env,env_ids):
    force,torque=push_state(env)
    env_ids=slice(None) if env_ids is None else env_ids
    if hasattr(env, '_seo_mirror_age'):
        env._seo_mirror_age[env_ids] = 0
    force[env_ids]=0.
    torque[env_ids]=0.
    env._seo_next_push[env_ids]=int(env.common_step_counter)+300
    env._seo_end_push[env_ids]=0
    robot=env.scene['robot']
    robot.write_external_wrench_to_sim(force[env_ids,None],torque[env_ids,None],
        env_ids=env_ids,body_ids=robot.find_bodies('Trunk')[0])

def randomize_default_pose(env,env_ids):
    data=env.scene['robot'].data
    data.default_joint_pos.add_(torch.empty_like(data.default_joint_pos).uniform_(-.03,.03))
    action=env.action_manager.get_term('joints')
    action._offset=data.default_joint_pos[:,joint_ids(env)].clone()

def excessive_velocity(env):
    data=env.scene['robot'].data
    return (data.root_link_lin_vel_b.square().sum(-1)+
        data.root_link_ang_vel_b.square().sum(-1))>50.

def velocity_kick(env,env_ids):
    robot=env.scene['robot']
    env_ids=slice(None) if env_ids is None else env_ids
    velocity=robot.data.root_link_vel_w[env_ids].clone()
    noise=torch.randn_like(velocity)
    noise[:,:3]*=.5
    noise[:,3:]*=.03
    robot.write_root_link_velocity_to_sim(velocity+noise,env_ids=env_ids)
