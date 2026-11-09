import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tyro
from seo_walk_mjlab.bootstrap import import_mjlab
import_mjlab()
from seo_walk_mjlab import TASK_ID,register
from mjlab.scripts.play import PlayConfig,run_play

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--vx',type=float,default=.2)
    parser.add_argument('--vy',type=float,default=0.)
    parser.add_argument('--yaw',type=float,default=0.)
    custom,remaining=parser.parse_known_args()
    if not (-1.5<=custom.vx<=3 and abs(custom.vy)<=1.5 and abs(custom.yaw)<=1.6):
        parser.error('Command outside the training range')
    register((custom.vx,custom.vy,custom.yaw))
    cfg=tyro.cli(PlayConfig,args=remaining,default=PlayConfig(num_envs=1,viewer='native'),
        config=(tyro.conf.AvoidSubcommands,))
    run_play(TASK_ID,cfg)
