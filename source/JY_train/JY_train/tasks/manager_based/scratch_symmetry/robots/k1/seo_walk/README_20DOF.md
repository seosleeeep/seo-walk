# seo_walk: 20DoF walking

- Model: K1_20dof.urdf, derived from INHA-RL sirl-worldmodel resources/K1/K1_22dof.urdf.
- Source blob and mesh hashes: assets/manifest.json.
- Only AAHead_yaw and Head_pitch were changed to fixed. Arms 8 + legs 12 remain revolute.
- The original remote K1_jy_locomotion model was unavailable, so identical leg geometry/inertias cannot be assumed.
- No training, simulation or hardware validation has been performed.

## Files to edit
- joint_config.py: 20 joint selection and nominal arm pose.
- robot_cfg.py: robot asset path, arm R14 actuator, gains and delay. Original leg actuators are deep-copied.
- tracking_env_cfg.py: 20 actions, 78 policy / 92 critic observations; joint regularizers and gain/default-position randomization include arms.
- symmetry.py: 20-joint left/right reflection and updated observation slices.
- env_cfg.py: task wiring and centroidal yaw momentum weight (-0.05).
- momentum_rewards.py: whole-body momentum cost, including arms when simulated.

Existing reward coefficients are unchanged. Sum-based joint penalties now cover 20 joints, so their measured contributions may increase.
Actions and joint observations retain Isaac Lab native simulator order (preserve_order=False). Use the resolved runtime joint names when exporting/deploying; do not assume the list order in joint_config.py.
12DoF checkpoints cannot be directly resumed into this 20DoF network. Train a new baseline; checkpoint dimension mapping is a separate change.
Arm gains (20 / 0.5), delay (0..18 physics steps), pose and momentum normalization are initial simulation settings, not validated hardware settings.

## Mesh setup on the GPU server
The full binary mesh set is NOT bundled: additional downloads were declined.
Set SEO_K1_MESH_DIR to the complete team's INHA-RL/resources/K1/meshes folder.
Without this variable, robot_cfg.py searches assets/meshes then booster_assets/robots/K1/meshes.
Missing files produce an explicit FileNotFoundError before simulation.
Do not substitute arbitrary meshes: collisions use some of these STL files too.
Model source: https://github.com/Inha-united-soccer/INHA-RL/tree/sirl-worldmodel/resources/K1

Example (replace the mesh path and choose an available GPU):
    conda activate env_isaaclab
    cd /path/to/INHA-WAlk-main
    export SEO_K1_MESH_DIR=/path/to/INHA-RL/resources/K1/meshes
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train.py --headless --device cuda:0 --task seo_walk_v0

This command has not been run. Start with a small-environment runtime check on the GPU server before full training.
The old 12DoF MuJoCo deploy configuration also needs 20DoF observation/action/joint-order updates; it cannot directly play this policy.
