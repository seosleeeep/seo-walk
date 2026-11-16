"""Use the installed official training runner with only Seo's task selected."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tyro
from sjy_vwalk_v2.bootstrap import import_mjlab
import_mjlab()
from sjy_vwalk_v2 import TASK_ID,register
from mjlab.scripts.train import TrainConfig,launch_training

if __name__=='__main__':
    register()
    cfg=tyro.cli(TrainConfig,default=TrainConfig.from_task(TASK_ID),
        config=(tyro.conf.AvoidSubcommands,))
    launch_training(TASK_ID,cfg)
