"""GPU integration check: dimensions, every reward, reset, and symmetry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from sjy_vwalk_v1.bootstrap import import_mjlab
import_mjlab()
from mjlab.envs import ManagerBasedRlEnv
from sjy_vwalk_v1.env_cfg import make_env_cfg
from sjy_vwalk_v1.robot import POLICY_JOINT_NAMES
from sjy_vwalk_v1 import mdp
from sjy_vwalk_v1.symmetry import mirror_policy_observations,mirror_critic_observations,mirror_actions

def main():
    cfg=make_env_cfg()
    cfg.scene.num_envs=16
    cfg.observations['actor'].enable_corruption=False
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
        obs,_=env.reset()
        assert obs['actor'].shape==(16,78),obs['actor'].shape
        assert obs['critic'].shape==(16,92),obs['critic'].shape
        assert len(env.scene['robot'].joint_names)==20
        assert not env.command_manager.get_term('motion').waypoint.any()
        defaults = env.scene['robot'].data.default_joint_pos[:, mdp.joint_ids(env)]
        torch.testing.assert_close(mirror_actions(defaults, env), defaults)
        assert env.action_manager.get_term('joints').target_names==POLICY_JOINT_NAMES
        for _ in range(320):
            action=torch.zeros(16,20,device=env.device)
            obs,reward,terminated,truncated,extras=env.step(action)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs['actor']).all()
        for name in mdp.REWARDS:
            values=mdp.reward(env,name)
            assert values.shape==(16,), (name,values.shape)
            assert torch.isfinite(values).all(),name
        for dim,func in [(78,mirror_policy_observations),(92,mirror_critic_observations),(20,mirror_actions)]:
            x=torch.randn(16,dim,device=env.device)
            torch.testing.assert_close(func(func(x,env),env),x)
        print('PASS: 20 actions / actor 78 / critic 92 / finite rewards / reset / mirror involution')
    finally:
        env.close()

if __name__=='__main__':
    main()
