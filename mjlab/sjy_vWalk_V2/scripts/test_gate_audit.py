import math
import unittest
from audit_speed_gate import audit_window,wrap


def trace(rate=0.):
    return [dict(time_s=i*.02,forward_m=i*.02,lateral_m=0.,heading_deg=0.,
        command_vx_mps=1.,body_vx_mps=1.,body_vy_mps=0.,body_wz_radps=rate) for i in range(1,501)]


class AuditTests(unittest.TestCase):
    def test_straight(self):
        self.assertTrue(audit_window(trace(),0,3)['original_pass'])

    def test_oscillation(self):
        rows=trace()
        for i,r in enumerate(rows):
            r['body_wz_radps']=.3 if i%2 else -.3
        result=audit_window(rows,0,3)
        self.assertFalse(result['original_pass'])
        self.assertTrue(result['candidate_mean_rms035_pass'])

    def test_large_oscillation_not_hidden_by_mean(self):
        rows=trace()
        for i,r in enumerate(rows):r['body_wz_radps']=.8 if i%2 else -.8
        result=audit_window(rows,0,3)
        self.assertTrue(result['candidate_mean_only_pass'])
        self.assertFalse(result['candidate_mean_rms050_pass'])

    def test_persistent_turn_and_heading(self):
        self.assertFalse(audit_window(trace(.3),0,3)['candidate_mean_only_pass'])
        rows=trace()
        for r in rows:r['heading_deg']=20.
        self.assertFalse(audit_window(rows,0,3)['candidate_mean_only_pass'])

    def test_reanchor(self):
        rows=trace()
        for r in rows:r['lateral_m']=.4
        self.assertFalse(audit_window(rows,0,3)['original_pass'])
        self.assertTrue(audit_window(rows,3,3)['original_pass'])
        self.assertAlmostEqual(wrap(math.radians(-358)),math.radians(2))

    def test_invalid_and_ineligible(self):
        with self.assertRaises(ValueError):audit_window(trace(),9,3)
        rows=trace()
        for r in rows:r['command_vx_mps']=.5;r['body_vx_mps']=.5
        self.assertFalse(audit_window(rows,0,3)['eligible'])


if __name__=='__main__':unittest.main()

