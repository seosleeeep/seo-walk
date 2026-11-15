"""Shared K1 MJCF, with the two head joints welded for a 20-action task."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from .joint_config import K1_ARM_DEFAULT_POS, K1_ARM_JOINT_NAMES

POLICY_JOINT_NAMES = [
    'ALeft_Shoulder_Pitch', 'ARight_Shoulder_Pitch',
    'Left_Hip_Pitch', 'Right_Hip_Pitch',
    'Left_Shoulder_Roll', 'Right_Shoulder_Roll',
    'Left_Hip_Roll', 'Right_Hip_Roll',
    'Left_Elbow_Pitch', 'Right_Elbow_Pitch',
    'Left_Hip_Yaw', 'Right_Hip_Yaw',
    'Left_Elbow_Yaw', 'Right_Elbow_Yaw',
    'Left_Knee_Pitch', 'Right_Knee_Pitch',
    'Left_Ankle_Pitch', 'Right_Ankle_Pitch',
    'Left_Ankle_Roll', 'Right_Ankle_Roll',
]
DEFAULT_POS = dict(K1_ARM_DEFAULT_POS)
for side in ('Left', 'Right'):
    DEFAULT_POS.update({f'{side}_{j}': v for j, v in
        [('Hip_Pitch', -.2), ('Hip_Roll', 0.), ('Hip_Yaw', 0.),
         ('Knee_Pitch', .4), ('Ankle_Pitch', -.2), ('Ankle_Roll', 0.)]})

def effort_limit(name):
    if name in K1_ARM_JOINT_NAMES:
        return 14.
    return next(v for k, v in [('Hip_Pitch',30.),('Hip_Roll',35.),
        ('Hip_Yaw',20.),('Knee_Pitch',40.),('Ankle_Pitch',20.),('Ankle_Roll',20.)]
        if name.endswith(k))

def transformed_xml(mesh_dir=None):
    mesh_dir = Path(mesh_dir or os.environ.get('SEO_K1_MESH_DIR',
        '/home/user/booster_assets/robots/K1/meshes')).expanduser().resolve()
    tree = ET.parse(Path(__file__).parent / 'assets/K1_source.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(mesh_dir))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'joint' and child.get('name') in ('AAHead_yaw','Head_pitch'):
                parent.remove(child)
    for tag in ('actuator','keyframe','sensor'):
        child = root.find(tag)
        if child is not None:
            root.remove(child)
    world = root.find('worldbody')
    for child in list(world):
        if child.tag in ('geom','light'):
            world.remove(child)
    missing = []
    for mesh in root.findall('./asset/mesh'):
        file = mesh_dir / Path(mesh.get('file')).name
        mesh.set('file', str(file))
        if not file.is_file():
            missing.append(str(file))
    if missing:
        raise FileNotFoundError('K1 mesh files missing: ' + ', '.join(missing))
    return ET.tostring(root, encoding='unicode')

def get_spec():
    return mujoco.MjSpec.from_string(transformed_xml())

def robot_cfg(play=False):
    actuators = []
    for name in POLICY_JOINT_NAMES:
        arm = name in K1_ARM_JOINT_NAMES
        foot = 'Ankle' in name
        actuators.append(IdealPdActuatorCfg(target_names_expr=(name,),
            stiffness=10. if arm else 30. if foot else 80.,
            damping=1. if arm else 2., effort_limit=effort_limit(name),
            delay_min_lag=0, delay_max_lag=0 if play else 18))
    return EntityCfg(spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.57), joint_pos=DEFAULT_POS),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators),
            soft_joint_pos_limit_factor=.9))

class OrderedJointPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        ids, names = self._entity.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
        if names != POLICY_JOINT_NAMES:
            raise ValueError(f'20-joint contract mismatch: {names}')
        return ids, names

@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return OrderedJointPositionAction(self, env)
