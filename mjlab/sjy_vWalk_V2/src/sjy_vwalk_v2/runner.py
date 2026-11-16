"""Checkpoint curriculum state alongside the standard runner's environment counter."""
from mjlab.rl import MjlabOnPolicyRunner


class CurriculumRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        motion = self.env.unwrapped.command_manager.get_term('motion')
        info = dict(infos or {})
        info['v2_speed_gate'] = motion.speed_gate.state_dict()
        return super().save(path, infos=info)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        info = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        motion = self.env.unwrapped.command_manager.get_term('motion')
        if not motion.cfg.play:
            if not info or 'v2_speed_gate' not in info:
                raise ValueError('V2 resume needs a V2 curriculum checkpoint; start fresh for V1 comparison.')
            motion.restore_gate(info['v2_speed_gate'])
        return info
