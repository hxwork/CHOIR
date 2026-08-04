"""Unit tests for interaction_motion_profile heuristics."""

import math
import unittest

import numpy as np

from interaction_motion_profile import (
    DEFAULT_THRESHOLDS,
    _classify_from_scores,
    _kabsch_2d_abs_angle_deg,
    build_interaction_motion_profile,
    mano_root_R_camera_series,
)


class InteractionMotionProfileTest(unittest.TestCase):
    def test_mano_root_series_and_profile(self):
        from scipy.spatial.transform import Rotation as SciR

        eye4 = np.eye(4, dtype=np.float64)
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        params = []
        for k in range(8):
            aa = (k * 4.0) * z
            params.append({
                "T_w2c": eye4.tolist(),
                "root_orient": aa.tolist(),
                "trans": [0.0, 0.0, 0.0],
            })
        R = mano_root_R_camera_series(np.array(params, dtype=object), 0, 8)
        self.assertIsNotNone(R)
        self.assertEqual(R.shape, (8, 3, 3))
        prof = build_interaction_motion_profile(
            0, 8, None, None, None, None, sampled_mano_params=np.array(params, dtype=object),
        )
        self.assertIsNotNone(prof.get("mano_global_root"))
        self.assertEqual(prof["suggested_mode"], "rotation_likely")

    def test_classify_cumulative_like_teapot_case(self):
        """84319-like: cumulative cue needs coherent 2D rotation quality."""
        mode = _classify_from_scores(
            1.4,
            2.5,
            2.35,
            10.5,
            None,
            36.3,
            8.0,
            None,
            None,
            None,
            None,
            DEFAULT_THRESHOLDS,
            cot_endpoint_quality={"n": 40, "rotation_gain": 0.72},
        )
        self.assertEqual(mode, "rotation_likely")

    def test_classify_83120_profile_as_translation_without_quality(self):
        mode = _classify_from_scores(
            1.2270862579183976,
            2.4912247626742343,
            4.171091531223093,
            36.772831490380355,
            None,
            42.94801902714392,
            19.642142397413057,
            1.6881441112328026,
            3.0906390180053136,
            59.08504389314808,
            31.21314863381537,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_84319_profile_as_rotation_with_endpoint_quality(self):
        mode = _classify_from_scores(
            1.3962985673299384,
            2.471128968167659,
            2.353667228484256,
            10.500028585022761,
            None,
            36.30376275057839,
            27.018993266884717,
            2.36744128842348,
            4.000013414622967,
            61.553473499010494,
            44.60294085207331,
            DEFAULT_THRESHOLDS,
            cot_endpoint_quality={"n": 40, "rotation_gain": 0.72},
        )
        self.assertEqual(mode, "rotation_likely")

    def test_classify_87218_profile_as_rotation_with_step_quality(self):
        mode = _classify_from_scores(
            1.4499911393076672,
            4.517173854558833,
            3.8644582944525165,
            12.779626998027027,
            None,
            56.549654432999006,
            0.9154529986733269,
            2.7816907417038093,
            4.432812735190055,
            108.48593892644857,
            14.250123376717537,
            DEFAULT_THRESHOLDS,
            cot_step_quality={"n": 39, "support_min": 40, "rotation_gain_p50": 0.55},
        )
        self.assertEqual(mode, "rotation_likely")

    def test_classify_20871_profile_as_translation(self):
        mode = _classify_from_scores(
            3.453040563058953,
            8.5378343005996,
            1.0706537828925689,
            3.4785736041051507,
            None,
            79.4199329503559,
            68.34934664361968,
            1.1993620320930707,
            2.2027567098596346,
            27.585326738140626,
            7.356294044846585,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_21179_profile_as_translation(self):
        mode = _classify_from_scores(
            1.8609266443436467,
            3.0334410967596512,
            3.530095923312902,
            9.948896112634088,
            None,
            42.80131281990388,
            37.33975727750443,
            1.8833256946989418,
            2.5735522860930278,
            43.31649097807566,
            28.464845356528993,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_24736_profile_as_translation(self):
        mode = _classify_from_scores(
            0.8792297588998206,
            1.6450019573010009,
            1.565757992744774,
            9.7839684478283,
            None,
            22.85997373139533,
            8.19230199293044,
            1.916932017585131,
            3.036025626421396,
            49.840232457213396,
            30.066433581425507,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_25012_profile_as_translation(self):
        mode = _classify_from_scores(
            0.1329006961389779,
            0.2942986866228406,
            3.9581557671597394,
            19.57348664669763,
            None,
            3.056716011196492,
            None,
            2.1328654268989626,
            3.4510193789901784,
            81.04888622216058,
            11.848809921700834,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_defaults_to_translation_when_only_mano_cumulative_is_large(self):
        mode = _classify_from_scores(
            1.0,
            2.0,
            None,
            None,
            None,
            None,
            None,
            2.0,
            3.0,
            34.0,
            10.0,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_classify_mask_noise_does_not_override_translation_motion(self):
        mode = _classify_from_scores(
            0.5,
            1.0,
            3.0,
            12.0,
            None,
            4.0,
            2.0,
            None,
            None,
            None,
            None,
            DEFAULT_THRESHOLDS,
        )
        self.assertEqual(mode, "translation_likely")

    def test_kabsch_pure_translation_zero_angle(self):
        p0 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
        p1 = p0 + np.array([3.0, -2.0])
        a = _kabsch_2d_abs_angle_deg(p0, p1)
        self.assertLess(a, 0.05)

    def test_kabsch_90deg_rotation(self):
        p0 = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        th = math.radians(90.0)
        r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        p1 = (r @ p0.T).T
        a = _kabsch_2d_abs_angle_deg(p0, p1)
        self.assertAlmostEqual(a, 90.0, delta=0.2)

    def test_build_profile_translation_like_cotracker(self):
        t, n = 20, 30
        tr = np.zeros((t, n, 2), dtype=np.float64)
        for i in range(t):
            tr[i, :, 0] = np.linspace(50, 150, n)
            tr[i, :, 1] = 80 + i * 0.3
        vis = np.ones((t, n), dtype=np.float64)
        prof = build_interaction_motion_profile(2, 18, tr, vis, amodal_masks=None, pnp_R_col_major=None)
        self.assertEqual(prof["interaction_sampled_half_open"], [2, 18])
        self.assertEqual(prof["suggested_mode"], "translation_likely")

    def test_build_profile_rotation_like_cotracker(self):
        t, n = 24, 40
        base = np.stack(
            [np.cos(np.linspace(0, 2 * math.pi, n, endpoint=False)),
             np.sin(np.linspace(0, 2 * math.pi, n, endpoint=False))],
            axis=1,
        ) * 40.0 + np.array([120.0, 100.0])
        tr = np.zeros((t, n, 2), dtype=np.float64)
        for i in range(t):
            ang = math.radians(i * 5.0)
            c, s = math.cos(ang), math.sin(ang)
            r = np.array([[c, -s], [s, c]], dtype=np.float64)
            tr[i] = (r @ base.T).T
        vis = np.ones((t, n), dtype=np.float64)
        thr = dict(DEFAULT_THRESHOLDS)
        thr["cot_mean_deg_rotation_likely"] = 3.0
        thr["cot_p90_deg_rotation_likely"] = 8.0
        prof = build_interaction_motion_profile(1, 22, tr, vis, None, None, thresholds=thr)
        self.assertEqual(prof["suggested_mode"], "rotation_likely")


if __name__ == "__main__":
    unittest.main()
