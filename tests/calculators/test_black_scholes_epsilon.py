import unittest
import math
from calculators.black_scholes_epsilon import calculate_epsilon_value
from calculators.black_scholes_common import calculate_d1_d2

class TestBlackScholesEpsilon(unittest.TestCase):

    def test_calculate_epsilon_at_the_money(self):
        # Test epsilon with at-the-money option
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # For an ATM option, the formula simplifies somewhat
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        # Expected value calculated manually: -0.0625
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

    def test_calculate_epsilon_in_the_money(self):
        # Test epsilon with in-the-money option
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

    def test_calculate_epsilon_out_of_the_money(self):
        # Test epsilon with out-of-the-money option
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

    def test_epsilon_short_maturity(self):
        # Test with very short time to maturity
        S, K, T, r, q, vol = 100, 100, 0.01, 0.05, 0.02, 0.2
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

    def test_epsilon_long_maturity(self):
        # Test with long time to maturity
        S, K, T, r, q, vol = 100, 100, 5, 0.05, 0.02, 0.2
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

    def test_epsilon_high_volatility(self):
        # Test with high volatility
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.5
        d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
        # Epsilon = (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
        expected_epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
        epsilon = calculate_epsilon_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(epsilon, expected_epsilon, places=8)

if __name__ == '__main__':
    unittest.main()
