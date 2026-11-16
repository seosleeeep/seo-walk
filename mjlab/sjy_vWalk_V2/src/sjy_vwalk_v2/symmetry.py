from __future__ import annotations
import torch
from tensordict import TensorDict
from .robot import POLICY_JOINT_NAMES
POLICY_OBS_DIM = 78
CRITIC_OBS_DIM = 92
ACTION_DIM = 20
COMMAND = slice(0, 10)
GAIT_PHASE = slice(10, 12)
PROJECTED_GRAVITY = slice(12, 15)
BASE_ANG_VEL = slice(15, 18)
JOINT_POS = slice(18, 38)
JOINT_VEL = slice(38, 58)
LAST_ACTION = slice(58, 78)
BASE_MASS_COM = slice(78, 82)
BASE_LIN_VEL = slice(82, 85)
BASE_HEIGHT = slice(85, 86)
PUSH_FORCE = slice(86, 89)
PUSH_TORQUE = slice(89, 92)

def _joint_permutation(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    selected_names = POLICY_JOINT_NAMES
    local_index = {name: index for index, name in enumerate(selected_names)}
    permutation = []
    signs = []
    for name in selected_names:
        counterpart = name.replace('Left_', '@_').replace('Right_', 'Left_').replace('@_', 'Right_')
        if counterpart not in local_index:
            raise ValueError(f'Mirror counterpart for {name!r} is missing: {counterpart!r}.')
        permutation.append(local_index[counterpart])
        signs.append(-1.0 if '_Roll' in name or '_Yaw' in name else 1.0)
    return (torch.tensor(permutation, dtype=torch.long, device=device), torch.tensor(signs, dtype=torch.float32, device=device))

def _mirror_joints(values: torch.Tensor, permutation: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(f'Expected a {ACTION_DIM}-D joint/action tensor, got {values.shape}.')
    return values[..., permutation] * signs

def _mirror_command(command: torch.Tensor) -> torch.Tensor:
    """Mirror [vx, vy, wz, frequency, foot-yaws, body targets, foot offsets]."""
    if command.shape[-1] != 10:
        raise ValueError(f'Expected a 10-D command, got {command.shape}.')
    out = command.clone()
    out[..., 0] = command[..., 0]
    out[..., 1] = -command[..., 1]
    out[..., 2] = -command[..., 2]
    out[..., 3] = command[..., 3]
    out[..., 4] = -command[..., 5]
    out[..., 5] = -command[..., 4]
    out[..., 6] = command[..., 6]
    out[..., 7] = -command[..., 7]
    out[..., 8] = -command[..., 8]
    out[..., 9] = command[..., 9]
    return out

def mirror_policy_observations(obs: torch.Tensor, env) -> torch.Tensor:
    """Apply sagittal reflection and a half-cycle gait-phase shift."""
    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(f'Expected {POLICY_OBS_DIM}-D policy observations, got {obs.shape}.')
    permutation, signs = _joint_permutation(env, obs.device)
    out = obs.clone()
    out[..., COMMAND] = _mirror_command(obs[..., COMMAND])
    phase = obs[..., GAIT_PHASE]
    out[..., GAIT_PHASE] = torch.where(obs[..., 3:4] == 0, phase, -phase)
    out[..., PROJECTED_GRAVITY] = obs[..., PROJECTED_GRAVITY] * obs.new_tensor([1.0, -1.0, 1.0])
    out[..., BASE_ANG_VEL] = obs[..., BASE_ANG_VEL] * obs.new_tensor([-1.0, 1.0, -1.0])
    out[..., JOINT_POS] = _mirror_joints(obs[..., JOINT_POS], permutation, signs)
    out[..., JOINT_VEL] = _mirror_joints(obs[..., JOINT_VEL], permutation, signs)
    out[..., LAST_ACTION] = _mirror_joints(obs[..., LAST_ACTION], permutation, signs)
    return out

def mirror_critic_observations(obs: torch.Tensor, env) -> torch.Tensor:
    """Mirror the 78-D common state and 14-D privileged simulator state."""
    if obs.shape[-1] != CRITIC_OBS_DIM:
        raise ValueError(f'Expected {CRITIC_OBS_DIM}-D critic observations, got {obs.shape}.')
    out = obs.clone()
    out[..., :POLICY_OBS_DIM] = mirror_policy_observations(obs[..., :POLICY_OBS_DIM], env)
    out[..., BASE_MASS_COM] = obs[..., BASE_MASS_COM] * obs.new_tensor([1.0, -1.0, 1.0, 1.0])
    out[..., BASE_LIN_VEL] = obs[..., BASE_LIN_VEL] * obs.new_tensor([1.0, -1.0, 1.0])
    out[..., BASE_HEIGHT] = obs[..., BASE_HEIGHT]
    out[..., PUSH_FORCE] = obs[..., PUSH_FORCE] * obs.new_tensor([1.0, -1.0, 1.0])
    out[..., PUSH_TORQUE] = obs[..., PUSH_TORQUE] * obs.new_tensor([-1.0, 1.0, -1.0])
    return out

def mirror_actions(actions: torch.Tensor, env) -> torch.Tensor:
    permutation, signs = _joint_permutation(env, actions.device)
    return _mirror_joints(actions, permutation, signs)

def augment(env, obs=None, actions=None):
    augmented_obs=None
    if obs is not None:
        mirrors={}
        for name,values in obs.items():
            if name=='actor':
                mirrors[name]=mirror_policy_observations(values,env)
            elif name=='critic':
                mirrors[name]=mirror_critic_observations(values,env)
            else:
                raise KeyError(f'Unknown observation group: {name}')
        mirrored=TensorDict(mirrors,batch_size=obs.batch_size,device=obs.device)
        augmented_obs=torch.cat([obs,mirrored],dim=0)
    augmented_actions=None if actions is None else torch.cat([actions,mirror_actions(actions,env)],dim=0)
    return augmented_obs,augmented_actions
