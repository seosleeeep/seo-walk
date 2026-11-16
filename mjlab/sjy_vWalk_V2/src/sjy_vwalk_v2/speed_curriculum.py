"""Performance gate state; standard-library only for independent testing."""
from dataclasses import dataclass, asdict


@dataclass
class SpeedGate:
    caps: tuple = (1.0, 1.5, 2.0, 2.5, 3.0)
    stage: int = 0
    stage_start_step: int = 0
    last_check_step: int = 0
    segments: int = 0
    successes: int = 0
    streak: int = 0
    last_rate: float = 0.0

    @property
    def cap(self):
        return self.caps[self.stage]

    def observe(self, count, successes):
        if not 0 <= successes <= count:
            raise ValueError('Invalid segment counts')
        self.segments += int(count)
        self.successes += int(successes)

    def check(self, step, rollout=24):
        # Non-overlapping windows; minimum dwell is 1000 PPO iterations.
        if step-self.last_check_step < 100*rollout:
            return None
        self.last_check_step = step
        count = self.segments
        rate = self.successes/count if count else 0.
        self.last_rate = rate
        qualified = count >= 200 and rate >= .8 and step-self.stage_start_step >= 1000*rollout
        self.streak = self.streak+1 if qualified else 0
        before = self.cap
        if self.streak >= 2 and self.stage < len(self.caps)-1:
            self.stage += 1
            self.stage_start_step = step
            self.streak = 0
        self.segments = self.successes = 0
        return dict(step=step, iteration=step//rollout, completed_segments=count,
                    success_rate=rate, cap_before=before, cap_after=self.cap,
                    promoted=self.cap != before)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        result = cls(**state)
        if tuple(result.caps) != cls().caps or not 0 <= result.stage < len(result.caps):
            raise ValueError('Invalid curriculum checkpoint')
        return result
