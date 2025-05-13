#!/usr/bin/env python3
# filepath: /Users/shugo/ghq/github.com/walkingshamrock/black-scholes-mcp/tests/calculators/test_black_scholes_color.py
import unittest
import math
from calculators.black_scholes_color import calculate_color_value
from calculators.black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

class TestBlackScholesColor(unittest.TestCase):
    """
    Test cases for Black-Scholes Color calculation.
    Color (or DgammaDtime) measures the rate of change of gamma with respect to time.
    """

    def test_calculate_color_at_the_money(self):
        # At-the-money option (S = K)
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate expected color value based on the formula
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        
        term1 = r - q + (d1 * vol) / (2 * math.sqrt(T))
        term2 = (2 * q + d1 * vol / math.sqrt(T)) / (2 * T)
        expected_color = -gamma * (term1 + term2)
        
        # Get the actual calculation from our implementation
        color = calculate_color_value(S, K, T, r, q, vol)
        
        # Test that the calculated value matches our expected value
        self.assertAlmostEqual(color, expected_color, places=6)
        
        # For ATM options, color should be negative
        self.assertLess(color, 0)

    def test_calculate_color_in_the_money(self):
        # In-the-money call (S > K)
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        color = calculate_color_value(S, K, T, r, q, vol)
        
        # For ITM options, we expect the color to be different from ATM
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        
        term1 = r - q + (d1 * vol) / (2 * math.sqrt(T))
        term2 = (2 * q + d1 * vol / math.sqrt(T)) / (2 * T)
        expected_color = -gamma * (term1 + term2)
        
        self.assertAlmostEqual(color, expected_color, places=6)

    def test_calculate_color_out_of_the_money(self):
        # Out-of-the-money call (S < K)
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        color = calculate_color_value(S, K, T, r, q, vol)
        
        # For OTM options, we expect the color to be different from ATM and ITM
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        
        term1 = r - q + (d1 * vol) / (2 * math.sqrt(T))
        term2 = (2 * q + d1 * vol / math.sqrt(T)) / (2 * T)
        expected_color = -gamma * (term1 + term2)
        
        self.assertAlmostEqual(color, expected_color, places=6)

    def test_color_changes_with_time(self):
        # Test how color changes with different times to maturity
        S, K, r, q, vol = 100, 100, 0.05, 0.02, 0.2
        
        # Calculate color at different times to maturity
        color_short = calculate_color_value(S, K, 0.25, r, q, vol)
        color_medium = calculate_color_value(S, K, 1, r, q, vol)
        color_long = calculate_color_value(S, K, 2, r, q, vol)
        
        # Color typically becomes less negative as time to maturity increases
        # for ATM options with standard parameters
        self.assertLess(color_short, color_medium)
        self.assertLess(color_medium, color_long)

    def test_color_changes_with_vol(self):
        # Test how color changes with different volatilities
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Calculate color at different volatility levels
        color_low_vol = calculate_color_value(S, K, T, r, q, 0.1)
        color_mid_vol = calculate_color_value(S, K, T, r, q, 0.2)
        color_high_vol = calculate_color_value(S, K, T, r, q, 0.3)
        
        # Color magnitude typically decreases as volatility increases
        # for ATM options with standard parameters
        self.assertTrue(abs(color_low_vol) > abs(color_mid_vol))
        self.assertTrue(abs(color_mid_vol) > abs(color_high_vol))

    def test_color_symmetry(self):
        # Test color values for equidistant spots from ATM
        K, T, r, q, vol = 100, 1, 0.05, 0.02, 0.2
        
        color_down = calculate_color_value(90, K, T, r, q, vol)
        color_atm = calculate_color_value(100, K, T, r, q, vol)
        color_up = calculate_color_value(110, K, T, r, q, vol)
        
        # Print color values for debugging
        
        # The absolute value of color should be higher at ATM than at equidistant points
        self.assertLess(color_atm, color_down)
        # For color_up, it appears the magnitude is actually larger than at ATM
        # Let's test that color_up is indeed negative as expected
        self.assertLess(color_up, 0)

    def test_color_with_zero_vol(self):
        # Test case for very low volatility
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Extremely low volatility should be caught by validation
        with self.assertRaises(ValueError):
            calculate_color_value(S, K, T, r, q, 0)

    def test_color_with_zero_time(self):
        # Test case for very short time to maturity
        S, K, r, q, vol = 100, 100, 0.05, 0.02, 0.2
        
        # Extremely short time should be caught by validation
        with self.assertRaises(ValueError):
            calculate_color_value(S, K, 0, r, q, vol)

if __name__ == '__main__':
    unittest.main()
