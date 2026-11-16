"""CPU-only gate tests; python scripts/test_speed_curriculum.py."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sjy_vwalk_v2.speed_curriculum import SpeedGate


class GateTests(unittest.TestCase):
    def test_no_time_only_promotion(self):
        gate = SpeedGate()
        for step in range(2400, 240001, 2400):
            gate.observe(200, 0)
            gate.check(step)
        self.assertEqual(gate.cap, 1.)

    def test_two_independent_windows(self):
        gate = SpeedGate()
        gate.observe(200, 200)
        gate.check(24000)
        self.assertEqual(gate.cap, 1.)
        gate.observe(199, 199)
        gate.check(26400)
        self.assertEqual(gate.streak, 0)
        for step in (28800,31200):
            gate.observe(200,160)
            gate.check(step)
        self.assertEqual(gate.cap, 1.5)
        self.assertEqual(gate.stage_start_step,31200)

    def test_dwell_and_ceiling(self):
        gate = SpeedGate()
        gate.observe(500,500)
        gate.check(2400)
        self.assertEqual(gate.streak,0)
        for step in range(4800,240001,2400):
            gate.observe(500,500)
            gate.check(step)
        self.assertEqual(gate.cap,3.)

    def test_checkpoint_roundtrip(self):
        gate = SpeedGate(stage=2, stage_start_step=100000, last_check_step=102400)
        gate.observe(72,60)
        restored = SpeedGate.from_state(gate.state_dict())
        self.assertEqual(restored.state_dict(),gate.state_dict())
        self.assertEqual(restored.check(102401),None)


if __name__ == '__main__':
    unittest.main()
