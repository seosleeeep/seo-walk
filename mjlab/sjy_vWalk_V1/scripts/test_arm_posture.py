"""CPU unit checks; no mjlab, meshes or GPU needed. Run from any directory."""
from pathlib import Path
import importlib.util
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'src/sjy_vwalk_v1'
spec = importlib.util.spec_from_file_location('arm_posture', PKG / 'arm_posture.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg = json.loads((PKG / 'reward_config.json').read_text())
params = cfg['arm_posture']['params']
assert cfg['arm_posture']['weight'] == -0.1
assert 'arm_deviation' not in cfg and 'arm_torque' not in cfg
speed = torch.linspace(0., 3., 301)
cmd = torch.zeros(301, 3)
cmd[:, 0] = speed
std = m.arm_posture_std(cmd, **params)
assert std.shape == (301, 8)
assert torch.all(std[1:] >= std[:-1] - 1e-7)
assert torch.allclose(std[:11], torch.full_like(std[:11], 0.05))
assert torch.allclose(std[100], torch.full((8,), 0.15))
assert torch.allclose(std[200], torch.tensor(params['run_std']))
assert torch.allclose(std[:, :4], std[:, 4:])
# Nonzero environment-specific reference, as used by randomized defaults.
default = torch.arange(8, dtype=torch.float32).expand(301, -1) * 0.03
assert torch.equal(m.arm_posture_cost(default, default, cmd, **params), torch.zeros(301))
cost = m.arm_posture_cost(default + 0.1, default, cmd, **params)
assert torch.all(cost[1:] <= cost[:-1] + 1e-5)
assert abs(cost[0].item() - 4.) < 1e-5
# Reflection maps corresponding joints, with roll/yaw signs flipped.
signs = torch.tensor([1., -1., 1., -1.] * 2)
perm = [4,5,6,7,0,1,2,3]
q = default + torch.linspace(-0.2, 0.25, 8)
mirror_cmd = cmd.clone(); mirror_cmd[:, 1:] *= -1
assert torch.allclose(m.arm_posture_cost(q, default, cmd, **params),
    m.arm_posture_cost(q[:,perm] * signs, default[:,perm] * signs, mirror_cmd, **params))
# Equal translational activity for forward/backward/lateral; yaw is included.
commands = torch.tensor([[0.5,0.,0.],[-0.5,0.,0.],[0.,0.5,0.],[0.,0.,0.5]])
assert torch.allclose(m.arm_posture_std(commands, **params),
                      m.arm_posture_std(commands, **params)[0].expand(4,-1))
# Check continuity at each knot.
for knot in [0.1,1.,2.]:
    c = torch.tensor([[knot-1e-6,0.,0.],[knot+1e-6,0.,0.]])
    pair = m.arm_posture_std(c, **params)
    assert torch.max((pair[0]-pair[1]).abs()) < 2e-6
print('PASS: config, defaults, 0.1 deadband, interpolation, monotonic cost, symmetry, yaw, continuity')
print('For 0.1 rad error in all 8 arm joints, weighted pre-dt costs:')
for i in [0,10,50,100,150,200,300]:
    print(f'  activity={speed[i].item():.2f}: {-0.1 * cost[i].item():.6f}')
