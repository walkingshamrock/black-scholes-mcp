import unittest
import math
from typing import Any, cast # Ensure Any and cast are imported
from calculators.black_scholes_common import norm_cdf, validate_inputs, calculate_d1_d2

class TestBlackScholesCommon(unittest.TestCase):

    def test_norm_cdf(self):
        self.assertAlmostEqual(norm_cdf(0), 0.5)
        self.assertAlmostEqual(norm_cdf(1.96), 0.9750021048517795, places=7)
        self.assertAlmostEqual(norm_cdf(-1.96), 0.024997895148220435, places=7)
        # Test with large positive and negative numbers
        self.assertAlmostEqual(norm_cdf(10), 1.0, places=7)
        self.assertAlmostEqual(norm_cdf(-10), 0.0, places=7)

    def test_validate_inputs_valid(self):
        # Test with a set of valid inputs
        try:
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type="call")
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type="put")
        except ValueError:
            self.fail("validate_inputs raised ValueError unexpectedly for valid inputs")

    def test_validate_inputs_invalid_S(self):
        with self.assertRaisesRegex(ValueError, r"Spot price \(S\) must be positive."):
            validate_inputs(S=0, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type="call")
        with self.assertRaisesRegex(ValueError, r"Spot price \(S\) must be positive."):
            validate_inputs(S=-100, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type="call")

    def test_validate_inputs_invalid_K(self):
        with self.assertRaisesRegex(ValueError, r"Strike price \(K\) must be positive."):
            validate_inputs(S=100, K=0, T=1, r=0.05, q=0.02, vol=0.2, option_type="call")
        with self.assertRaisesRegex(ValueError, r"Strike price \(K\) must be positive."):
            validate_inputs(S=100, K=-100, T=1, r=0.05, q=0.02, vol=0.2, option_type="call")

    def test_validate_inputs_invalid_T(self):
        with self.assertRaisesRegex(ValueError, r"Time to maturity \(T\) must be positive."):
            validate_inputs(S=100, K=100, T=0, r=0.05, q=0.02, vol=0.2, option_type="call")
        with self.assertRaisesRegex(ValueError, r"Time to maturity \(T\) must be positive."):
            validate_inputs(S=100, K=100, T=-1, r=0.05, q=0.02, vol=0.2, option_type="call")

    def test_validate_inputs_invalid_vol(self):
        with self.assertRaisesRegex(ValueError, r"Volatility \(vol\) must be positive."):
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=0, option_type="call")
        with self.assertRaisesRegex(ValueError, r"Volatility \(vol\) must be positive."):
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=-0.2, option_type="call")

    def test_validate_inputs_invalid_option_type(self):
        with self.assertRaisesRegex(ValueError, r"Option type must be 'call' or 'put'."):
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type=cast(Any, "CALL"))
        with self.assertRaisesRegex(ValueError, r"Option type must be 'call' or 'put'."):
            validate_inputs(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, option_type=cast(Any, "other"))

    def test_calculate_d1_d2_standard(self):
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        # Expected: d1 = (ln(100/100) + (0.05 - 0.02 + 0.5 * 0.2^2) * 1) / (0.2 * sqrt(1)) = 0.25
        # Expected: d2 = d1 - 0.2 * sqrt(1) = 0.05
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        self.assertAlmostEqual(d1, 0.25, places=7)
        self.assertAlmostEqual(d2, 0.05, places=7)

    def test_calculate_d1_d2_in_the_money_call(self):
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        # d1 = (ln(1.1) + (0.03 + 0.02)) / 0.2 = (0.0953101798 + 0.05) / 0.2 = 0.1453101798 / 0.2 = 0.726550899
        # d2 = d1 - 0.2 = 0.526550899
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        self.assertAlmostEqual(d1, 0.726550899, places=7)
        self.assertAlmostEqual(d2, 0.526550899, places=7)

    def test_calculate_d1_d2_out_of_money_call(self):
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        # d1 = (ln(0.9) + (0.03 + 0.02)) / 0.2 = (-0.1053605156 + 0.05) / 0.2 = -0.0553605156 / 0.2 = -0.276802578
        # d2 = d1 - 0.2 = -0.476802578
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        self.assertAlmostEqual(d1, -0.276802578, places=7)
        self.assertAlmostEqual(d2, -0.476802578, places=7)
        
    def test_calculate_d1_d2_zero_time_to_maturity(self):
        # With T->0, if S > K, d1 and d2 -> inf. If S < K, d1 and d2 -> -inf.
        # If S = K, d1 and d2 -> 0.
        # Our validate_inputs prevents T=0. Let's test with very small T.
        r, q, vol = 0.05, 0.02, 0.2
        T_small = 1e-9

        # Case 1: S = K (At-the-money)
        S_atm, K_atm = 100, 100
        d1_atm, d2_atm = calculate_d1_d2(S_atm, K_atm, T_small, r, q, vol)
        self.assertAlmostEqual(d1_atm, 0.0, places=4, msg="d1 for ATM with T_small should be close to 0") # Adjusted precision to 4
        self.assertAlmostEqual(d2_atm, 0.0, places=4, msg="d2 for ATM with T_small should be close to 0") # Adjusted precision to 4

        # Case 2: S > K (In-the-money call / Out-of-the-money put)
        S_itm, K_itm = 101, 100 
        d1_itm, d2_itm = calculate_d1_d2(S_itm, K_itm, T_small, r, q, vol)
        self.assertTrue(d1_itm > 1000, "d1 for ITM with T_small should be large positive") # Increased threshold for more certainty
        self.assertTrue(d2_itm > 1000, "d2 for ITM with T_small should be large positive")

        # Case 3: S < K (Out-of-the-money call / In-the-money put)
        S_otm, K_otm = 99, 100
        d1_otm, d2_otm = calculate_d1_d2(S_otm, K_otm, T_small, r, q, vol)
        self.assertTrue(d1_otm < -1000, "d1 for OTM with T_small should be large negative") # Increased threshold
        self.assertTrue(d2_otm < -1000, "d2 for OTM with T_small should be large negative")

    def test_calculate_d1_d2_zero_volatility(self):
        # validate_inputs prevents vol=0. Test with very small vol.
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        vol_small = 1e-9
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol_small)
        # (r - q + vol^2/2) = 0.05 - 0.02 + (1e-9)^2/2 approx 0.03 (positive)
        # d1 = (ln(S/K) + (r - q + vol^2/2)T) / (vol * sqrt(T))
        # d1 = (0 + 0.03 * 1) / (1e-9 * 1) = 0.03 / 1e-9 = 3e7 (large positive)
        self.assertTrue(d1 > 1e6) # Heuristic check for very large positive
        self.assertTrue(d2 > 1e6) # Heuristic check for very large positive

        S_itm, K_itm = 101, 100 # S > K, ln(S/K) is positive
        d1_itm, d2_itm = calculate_d1_d2(S_itm, K_itm, T, r, q, vol_small)
        self.assertTrue(d1_itm > 1e6)
        self.assertTrue(d2_itm > 1e6)

        S_otm, K_otm = 99, 100 # S < K, ln(S/K) is negative
        # d1 = (ln(0.99) + 0.03) / 1e-9 = (-0.01005 + 0.03) / 1e-9 = 0.01995 / 1e-9 = 1.995e7
        # This case is tricky, if (r-q+vol^2/2)T dominates ln(S/K)
        # Let's use S=90, K=100. ln(0.9) = -0.10536
        # d1 = (-0.10536 + 0.03) / 1e-9 = -0.07536 / 1e-9 = -7.536e7 (large negative)
        S_deep_otm, K_deep_otm = 90, 100
        d1_deep_otm, d2_deep_otm = calculate_d1_d2(S_deep_otm, K_deep_otm, T, r, q, vol_small)
        self.assertTrue(d1_deep_otm < -1e6)
        self.assertTrue(d2_deep_otm < -1e6)

if __name__ == '__main__':
    unittest.main()
