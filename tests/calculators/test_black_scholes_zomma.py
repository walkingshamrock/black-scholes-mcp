#!/usr/bin/env python3
# filepath: /Users/shugo/ghq/github.com/walkingshamrock/black-scholes-mcp/tests/calculators/test_black_scholes_zomma.py
import unittest
import math
from calculators.black_scholes_zomma import calculate_zomma_value
from calculators.black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

class TestBlackScholesZomma(unittest.TestCase):
    """
    Test cases for Black-Scholes Zomma calculation.
    Zomma measures the rate of change of gamma with respect to changes in volatility.
    It is the third derivative of the option price with respect to the underlying price.
    """

    def test_calculate_zomma_at_the_money(self):
        # At-the-money option (S = K)
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate expected zomma value based on the formula
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        expected_zomma = (d1 * d2 - 1) * gamma / vol
        
        # Get the actual calculation from our implementation
        zomma = calculate_zomma_value(S, K, T, r, q, vol)
        
        # Test that the calculated value matches our expected value
        self.assertAlmostEqual(zomma, expected_zomma, places=6)
        
        # Additional verification: for ATM options, zomma should be negative
        # when d1*d2 < 1, which is typically the case for reasonable parameters
        if d1 * d2 < 1:
            self.assertLess(zomma, 0)

    def test_calculate_zomma_in_the_money(self):
        # In-the-money call (S > K)
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        zomma = calculate_zomma_value(S, K, T, r, q, vol)
        
        # For ITM options, we expect the zomma to be different from ATM
        # We're testing the magnitude rather than the exact value
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        expected_zomma = (d1 * d2 - 1) * gamma / vol
        
        self.assertAlmostEqual(zomma, expected_zomma, places=6)

    def test_calculate_zomma_out_of_the_money(self):
        # Out-of-the-money call (S < K)
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        zomma = calculate_zomma_value(S, K, T, r, q, vol)
        
        # For OTM options, we expect the zomma to be different from ATM and ITM
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
        gamma_denominator = S * vol * math.sqrt(T)
        gamma = gamma_numerator / gamma_denominator
        expected_zomma = (d1 * d2 - 1) * gamma / vol
        
        self.assertAlmostEqual(zomma, expected_zomma, places=6)

    def test_zomma_as_volatility_changes(self):
        # Test how zomma changes as volatility changes
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Calculate zomma at different volatility levels
        zomma_low_vol = calculate_zomma_value(S, K, T, r, q, 0.1)
        zomma_mid_vol = calculate_zomma_value(S, K, T, r, q, 0.2)
        zomma_high_vol = calculate_zomma_value(S, K, T, r, q, 0.3)
        
        # Zomma should be more negative as volatility decreases for ATM options
        self.assertLess(zomma_low_vol, zomma_mid_vol)
        self.assertLess(zomma_mid_vol, zomma_high_vol)

    def test_zomma_symmetry_around_atm(self):
        # Test the symmetry of zomma around the ATM point
        S_base, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate zomma at equidistant points from ATM
        S_up = 110
        S_down = 90
        
        zomma_up = calculate_zomma_value(S_up, K, T, r, q, vol)
        zomma_atm = calculate_zomma_value(S_base, K, T, r, q, vol)
        zomma_down = calculate_zomma_value(S_down, K, T, r, q, vol)
        
        # Zomma should be more negative at ATM than at equidistant OTM or ITM points
        self.assertLess(zomma_atm, zomma_up)
        self.assertLess(zomma_atm, zomma_down)

    def test_zomma_time_impact(self):
        # Test how time to maturity affects zomma
        S, K, r, q, vol = 100, 100, 0.05, 0.02, 0.2
        
        # Calculate zomma at different times to maturity
        zomma_short = calculate_zomma_value(S, K, 0.25, r, q, vol)
        zomma_medium = calculate_zomma_value(S, K, 1, r, q, vol)
        zomma_long = calculate_zomma_value(S, K, 2, r, q, vol)
        
        # For ATM options, zomma typically becomes less negative as time increases
        self.assertLess(zomma_short, zomma_medium)
        self.assertLess(zomma_medium, zomma_long)

    def test_zomma_with_zero_volatility(self):
        # Test case for very low volatility
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Extremely low volatility should cause an error
        with self.assertRaises(ValueError):
            calculate_zomma_value(S, K, T, r, q, 0)

if __name__ == '__main__':
    unittest.main()
