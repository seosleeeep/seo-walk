"""Own K1 vv3 MuJoCo task. Does not import lsh_walk or inha_mjlab."""
TASK_ID='sjy_vWalk_V2'

def register(velocity=(0.,0.,0.),distance=None):
    from mjlab.tasks.registry import register_mjlab_task
    from .env_cfg import make_env_cfg
    from .ppo_cfg import make_ppo_cfg
    from .runner import CurriculumRunner
    register_mjlab_task(TASK_ID,env_cfg=make_env_cfg(),
        play_env_cfg=make_env_cfg(True,velocity,distance),rl_cfg=make_ppo_cfg(),runner_cls=CurriculumRunner)
