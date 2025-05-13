import unittest
import math
from calculators.black_scholes_vanna import calculate_vanna_value
from calculators.black_scholes_common import calculate_d1_d2, norm_pdf

class TestBlackScholesVanna(unittest.TestCase):

    def test_calculate_vanna_at_the_money(self):
        # Test vanna with at-the-money option
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

    def test_calculate_vanna_in_the_money(self):
        # Test vanna with in-the-money option
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

    def test_calculate_vanna_out_of_the_money(self):
        # Test vanna with out-of-the-money option
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

    def test_vanna_short_maturity(self):
        # Test with very short time to maturity
        S, K, T, r, q, vol = 100, 100, 0.01, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

    def test_vanna_long_maturity(self):
        # Test with long time to maturity
        S, K, T, r, q, vol = 100, 100, 5, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

    def test_vanna_high_volatility(self):
        # Test with high volatility
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.5
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        # Vanna = -e^(-qt) * N'(d1) * d2 / vol
        expected_vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
        vanna = calculate_vanna_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(vanna, expected_vanna, places=8)

if __name__ == '__main__':
    unittest.main()
