from dataclasses import dataclass,field
from mjlab.rl import RslRlOnPolicyRunnerCfg,RslRlPpoAlgorithmCfg,RslRlModelCfg

@dataclass
class SymmetricPpoCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict = field(default_factory=lambda:{
        'use_data_augmentation':True,'use_mirror_loss':True,'mirror_loss_coeff':.005,
        'data_augmentation_func':'seo_walk_mjlab.symmetry:augment'})

def make_ppo_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.15,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(512,256,128),activation='elu',obs_normalization=True),
        algorithm=SymmetricPpoCfg(num_learning_epochs=5,num_mini_batches=4,
            learning_rate=2.5e-6,schedule='adaptive',entropy_coef=.005),
        num_steps_per_env=24,max_iterations=30000,save_interval=100,
        experiment_name='seo_walk_vv0_mjlab',run_name='vv0_mj_first',
        logger='tensorboard',upload_model=False,clip_actions=None)
