"""No GPU allocation: dependency/API configuration and real K1 mesh compilation."""
import sys
from pathlib import Path
from importlib.metadata import version
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sjy_vwalk_v1.bootstrap import import_mjlab
import_mjlab()
import mujoco
from sjy_vwalk_v1.robot import transformed_xml,POLICY_JOINT_NAMES
from sjy_vwalk_v1.env_cfg import make_env_cfg
from sjy_vwalk_v1.ppo_cfg import make_ppo_cfg

for package in ('mjlab','mujoco','mujoco-warp','rsl-rl-lib'):
    print(package,version(package))
model=mujoco.MjModel.from_xml_string(transformed_xml())
names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
assert set(names)-set(POLICY_JOINT_NAMES)=={'world_joint'},names
assert model.nq==27 and model.nv==26,(model.nq,model.nv)
cfg=make_env_cfg()
agent=make_ppo_cfg()
assert 'straight_pose_mirror' in cfg.rewards
assert 'arm_torque' not in cfg.rewards
assert cfg.rewards['arm_posture'].weight==-.1
assert agent.algorithm.num_learning_epochs==5
print('PASS: real K1 mesh compilation; head welded; 20 hinge joints; arm posture enabled')
