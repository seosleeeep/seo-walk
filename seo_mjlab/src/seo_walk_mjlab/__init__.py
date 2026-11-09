"""Own K1 vv0 MuJoCo task. Does not import lsh_walk or inha_mjlab."""
TASK_ID='seo_walk_vv0_mjlab'

def register(velocity=(.2,0.,0.)):
    from mjlab.tasks.registry import register_mjlab_task
    from .env_cfg import make_env_cfg
    from .ppo_cfg import make_ppo_cfg
    register_mjlab_task(TASK_ID,env_cfg=make_env_cfg(),
        play_env_cfg=make_env_cfg(True,velocity),rl_cfg=make_ppo_cfg())
