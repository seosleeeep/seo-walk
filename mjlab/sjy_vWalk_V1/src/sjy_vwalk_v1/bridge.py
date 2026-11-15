"""Explicit adapter for the original reward math; no Isaac imports."""
from dataclasses import dataclass, field
from types import SimpleNamespace
import re
import torch
from .robot import effort_limit

@dataclass
class Selection:
    name: str
    joint_names: object = None
    body_names: object = None
    joint_ids: object = field(default_factory=lambda: slice(None))
    body_ids: object = field(default_factory=lambda: slice(None))

def match_names(patterns, names):
    if patterns is None:
        return slice(None)
    patterns = [patterns] if isinstance(patterns, str) else patterns
    ids = []
    for pattern in patterns:
        found = [i for i,n in enumerate(names) if re.fullmatch(pattern,n)]
        if not found:
            raise ValueError(f'No match for {pattern}: {names}')
        ids.extend(i for i in found if i not in ids)
    return ids

def resolve(value, env):
    if not isinstance(value, dict) or 'selection' not in value:
        return value
    selection = Selection(value['selection'], value.get('joint_names'),value.get('body_names'))
    if selection.name == 'contact_forces':
        names = [n.removeprefix('robot/') for n in env.scene['contact_forces'].primary_names]
        selection.body_ids = match_names(selection.body_names,names)
    else:
        robot = env.scene[selection.name]
        selection.joint_ids = match_names(selection.joint_names,robot.joint_names)
        selection.body_ids = match_names(selection.body_names,robot.body_names)
    return selection

class DataProxy:
    aliases = {'root_pos_w':'root_link_pos_w', 'root_quat_w':'root_link_quat_w',
        'root_lin_vel_b':'root_link_lin_vel_b', 'root_ang_vel_b':'root_link_ang_vel_b',
        'root_lin_vel_w':'root_link_lin_vel_w', 'body_pos_w':'body_link_pos_w',
        'body_quat_w':'body_link_quat_w', 'body_lin_vel_w':'body_link_lin_vel_w',
        'applied_torque':'qfrc_actuator'}
    def __init__(self,env):
        self.env=env
        self.robot=env.scene['robot']
        self.previous_linear=None
        self.previous_angular=None
        self.last_step=-1
        self.limits=torch.tensor([effort_limit(n) for n in self.robot.joint_names],
            device=env.device)[None].expand(env.num_envs,-1)
    def acceleration(self):
        step=int(self.env.common_step_counter)
        if step != self.last_step:
            lin=self.robot.data.body_link_lin_vel_w
            ang=self.robot.data.body_link_ang_vel_w
            if self.previous_linear is None:
                self.linear=torch.zeros_like(lin)
                self.angular=torch.zeros_like(ang)
            else:
                self.linear=(lin-self.previous_linear)/self.env.step_dt
                self.angular=(ang-self.previous_angular)/self.env.step_dt
                reset=self.env.episode_length_buf<=1
                self.linear[reset]=0.
                self.angular[reset]=0.
            self.previous_linear=lin.clone()
            self.previous_angular=ang.clone()
            self.last_step=step
    def __getattr__(self,name):
        if name in ('soft_joint_effort_limits','joint_effort_limits'):
            return self.limits
        if name in ('body_lin_acc_w','body_ang_acc_w'):
            self.acceleration()
            return self.linear if name=='body_lin_acc_w' else self.angular
        return getattr(self.robot.data,self.aliases.get(name,name))

class SensorDataProxy:
    def __init__(self,sensor):
        self.sensor=sensor
    @property
    def net_forces_w_history(self):
        values=self.sensor.data.force_history
        if values is None:
            raise RuntimeError('Contact force history must be enabled')
        return values.permute(0,2,1,3)

class SceneProxy:
    def __init__(self,env):
        self.robot=SimpleNamespace(data=DataProxy(env))
        self.sensors={'contact_forces':SimpleNamespace(
            data=SensorDataProxy(env.scene['contact_forces']))}
    def __getitem__(self,name):
        return self.robot if name=='robot' else self.sensors[name]

class EnvProxy:
    def __init__(self,env):
        object.__setattr__(self,'raw',env)
        object.__setattr__(self,'scene',SceneProxy(env))
    def __getattr__(self,name):
        return getattr(self.raw,name)
    def __setattr__(self,name,value):
        setattr(self.raw,name,value)

def bridge(env):
    if not hasattr(env,'_seo_reward_bridge'):
        env._seo_reward_bridge=EnvProxy(env)
    return env._seo_reward_bridge
