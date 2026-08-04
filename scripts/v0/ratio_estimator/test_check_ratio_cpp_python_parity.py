#!/usr/bin/env python3

import unittest

import pandas as pd

from check_ratio_cpp_python_parity import check_parity


class ParityTest(unittest.TestCase):
    def frame(self):
        # n=2, ell=3, s=1, error=0 -> kappa=1; dot=1 -> rho=.5
        # D=4+9-6=7; q=2 -> ratio LB=5.
        return pd.DataFrame([{
            "current_squared_distance": 4.0,
            "threshold": 5.0,
            "current_lb": 1.0,
            "current_lb_valid": 1,
            "edge_length": 3.0,
            "direction_error": 0.0,
            "reconstruction_norm": 1.0,
            "x_dot_reconstruction": 1.0,
            "kappa_meta": 1.0,
            "ratio_eligible": 1,
            "rho_hat_raw": 0.5,
            "rho_hat_ratio": 0.5,
            "ratio_estimated_squared_distance": 7.0,
            "ratio_quantile": 2.0,
            "ratio_lb": 5.0,
            "ratio_effective_lb": 5.0,
            "ratio_used_current_fallback": 0,
            "ratio_would_prune": 0,
        }])

    def test_strict_equality_does_not_prune(self):
        self.assertEqual(check_parity(self.frame(), 1e-12, 1e-12)["status"], "PASS")

    def test_detects_numeric_drift(self):
        frame = self.frame()
        frame.loc[0, "ratio_lb"] = 5.1
        self.assertEqual(check_parity(frame, 1e-12, 1e-12)["status"], "FAIL")

    def test_ineligible_without_current_bound_does_not_claim_fallback(self):
        frame = self.frame()
        frame.loc[0, "ratio_eligible"] = 0
        frame.loc[0, "current_lb_valid"] = 0
        frame.loc[0, "ratio_effective_lb"] = 0.0
        frame.loc[0, "ratio_used_current_fallback"] = 0
        frame.loc[0, "ratio_would_prune"] = 0
        self.assertEqual(check_parity(frame, 1e-12, 1e-12)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
