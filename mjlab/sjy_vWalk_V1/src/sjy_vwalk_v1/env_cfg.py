"""Independent flat-ground vv0 port for the supplied mjlab 1.6 API."""
import math
from mjlab.envs import ManagerBasedRlEnvCfg, mdp as builtin
from mjlab.envs.mdp import dr
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.viewer import ViewerConfig
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from .robot import robot_cfg, OrderedJointPositionActionCfg, POLICY_JOINT_NAMES
from .commands import WalkCommandCfg
from . import mdp

def common_observations(noise):
    terms={}
    for name,func,amplitude in [
        ('command',mdp.command,0.),('gait_phase',mdp.gait_phase,0.),
        ('projected_gravity',mdp.gravity,.05),('base_ang_vel',mdp.angular_velocity,.2),
        ('joint_pos',mdp.joint_position,.01),('joint_vel',mdp.joint_velocity,.5),
        ('actions',mdp.actions,0.)]:
        terms[name]=ObservationTermCfg(func=func,noise=UniformNoiseCfg(
            n_min=-amplitude,n_max=amplitude) if noise and amplitude else None)
    return terms

def make_env_cfg(play=False,velocity=(0.,0.,0.),distance=None):
    trunk=SceneEntityCfg('robot',body_names=('Trunk',))
    events={
        'reset_scene':EventTermCfg(func=builtin.reset_scene_to_default,mode='reset'),
        'reset_root':EventTermCfg(func=builtin.reset_root_state_uniform,mode='reset',params={
            'pose_range':{} if play else {'x':(-.2,.2),'y':(-.2,.2),'yaw':(-math.pi,math.pi)},
            'velocity_range':{} if play else {'x':(-.1,.1),'y':(-.1,.1),'yaw':(-.1,.1)}}),
        'reset_joints':EventTermCfg(func=builtin.reset_joints_by_offset,mode='reset',params={
            'position_range':(0.,0.) if play else (-.05,.05),
            'velocity_range':(0.,0.) if play else (-.1,.1)}),
        'reset_port':EventTermCfg(func=mdp.reset_port_state,mode='reset'),
    }
    if not play:
        events.update({
            'friction':EventTermCfg(func=dr.geom_friction,mode='startup',params={
                'ranges':(.1,2.),'axes':[0],'shared_random':True}),
            'mass':EventTermCfg(func=dr.body_mass,mode='startup',params={
                'ranges':(.8,1.2),'asset_cfg':trunk,'operation':'scale'}),
            'com':EventTermCfg(func=dr.body_com_offset,mode='startup',params={
                'ranges':(-.05,.05),'asset_cfg':trunk,'operation':'add'}),
            'default_pose':EventTermCfg(func=mdp.randomize_default_pose,mode='startup'),
            'pd_gains':EventTermCfg(func=dr.pd_gains,mode='reset',params={
                'kp_range':(.9,1.1),'kd_range':(.9,1.1)}),
            'forces':EventTermCfg(func=mdp.disturbance,mode='step'),
            'velocity_kick':EventTermCfg(func=mdp.velocity_kick,mode='interval',
                interval_range_s=(4.,4.)),
        })
    critic=common_observations(False)
    critic['privileged']=ObservationTermCfg(func=mdp.privileged)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=1 if play else 1024,env_spacing=2.5,
            terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':robot_cfg(play)},
            sensors=(ContactSensorCfg(name='contact_forces',primary=ContactMatch(
                mode='body',pattern='.*',entity='robot'),fields=('found','force'),
                reduce='netforce',num_slots=1,history_length=3,track_air_time=True),)),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=.002),contact_sensor_maxmatch=500),
        decimation=10,episode_length_s=30.,
        observations={
            'actor':ObservationGroupCfg(terms=common_observations(not play),
                concatenate_terms=True,enable_corruption=not play),
            'critic':ObservationGroupCfg(terms=critic,concatenate_terms=True,enable_corruption=False)},
        actions={'joints':OrderedJointPositionActionCfg(entity_name='robot',
            actuator_names=POLICY_JOINT_NAMES,scale={'ALeft_Shoulder_Pitch': 0.3, 'ARight_Shoulder_Pitch': 0.3, 'Left_Hip_Pitch': 1.0, 'Right_Hip_Pitch': 1.0, 'Left_Shoulder_Roll': 0.05, 'Right_Shoulder_Roll': 0.05, 'Left_Hip_Roll': 1.0, 'Right_Hip_Roll': 1.0, 'Left_Elbow_Pitch': 0.3, 'Right_Elbow_Pitch': 0.3, 'Left_Hip_Yaw': 1.0, 'Right_Hip_Yaw': 1.0, 'Left_Elbow_Yaw': 0.05, 'Right_Elbow_Yaw': 0.05, 'Left_Knee_Pitch': 1.0, 'Right_Knee_Pitch': 1.0, 'Left_Ankle_Pitch': 1.0, 'Right_Ankle_Pitch': 1.0, 'Left_Ankle_Roll': 1.0, 'Right_Ankle_Roll': 1.0},use_default_offset=True)},
        commands={'motion':WalkCommandCfg(resampling_time_range=(2.,4.),
            play=play,play_velocity_command=velocity,distance=distance)},events=events,
        rewards={"straight_pose_mirror": RewardTermCfg(func=mdp.straight_pose_mirror, weight=-0.5)} | {name:RewardTermCfg(func=mdp.reward,weight=entry['weight'],
            params={'term_name':name}) for name,entry in mdp.REWARDS.items()},
        terminations={
            'time_out':TerminationTermCfg(func=builtin.time_out,time_out=True),
            'base_height':TerminationTermCfg(func=builtin.root_height_below_minimum,
                params={'minimum_height':.35}),
            'root_velocity':TerminationTermCfg(func=mdp.excessive_velocity)},
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name='robot',body_name='Trunk',distance=2.,elevation=-15.,azimuth=90.))
