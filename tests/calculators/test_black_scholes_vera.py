#!/usr/bin/env python3
# filepath: /Users/shugo/ghq/github.com/walkingshamrock/black-scholes-mcp/tests/calculators/test_black_scholes_vera.py
import unittest
import math
from calculators.black_scholes_vera import calculate_vera_value
from calculators.black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

class TestBlackScholesVera(unittest.TestCase):
    """
    Test cases for Black-Scholes Vera calculation.
    Vera (or DrhoDvol) measures the rate of change of rho with respect to volatility.
    """

    def test_calculate_vera_at_the_money(self):
        # At-the-money option (S = K)
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate expected vera value based on the formula for a call
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        expected_vera = -K * T * math.exp(-r * T) * norm_pdf(d2) * d1 / (vol * math.sqrt(T))
        
        # Get the actual calculation from our implementation
        vera = calculate_vera_value(S, K, T, r, q, vol)
        
        # Test that the calculated value matches our expected value
        self.assertAlmostEqual(vera, expected_vera, places=6)

    def test_calculate_vera_in_the_money(self):
        # In-the-money call (S > K)
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate expected vera value
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        expected_vera = -K * T * math.exp(-r * T) * norm_pdf(d2) * d1 / (vol * math.sqrt(T))
        
        # Get the actual calculation
        vera = calculate_vera_value(S, K, T, r, q, vol)
        
        # Test that the calculated value matches our expected value
        self.assertAlmostEqual(vera, expected_vera, places=6)

    def test_calculate_vera_out_of_the_money(self):
        # Out-of-the-money call (S < K)
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate expected vera value
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        expected_vera = -K * T * math.exp(-r * T) * norm_pdf(d2) * d1 / (vol * math.sqrt(T))
        
        # Get the actual calculation
        vera = calculate_vera_value(S, K, T, r, q, vol)
        
        # Test that the calculated value matches our expected value
        self.assertAlmostEqual(vera, expected_vera, places=6)

    def test_vera_sign_for_call(self):
        # Test the sign of vera for different moneyness levels for a call option
        K, T, r, q, vol = 100, 1, 0.05, 0.02, 0.2
        
        # For deep ITM call (S >> K)
        vera_itm = calculate_vera_value(130, K, T, r, q, vol)
        # For ATM call
        vera_atm = calculate_vera_value(100, K, T, r, q, vol)
        # For deep OTM call (S << K)
        vera_otm = calculate_vera_value(70, K, T, r, q, vol)
        
        # For calls, vera should be negative for deep ITM options due to positive d1
        self.assertLess(vera_itm, 0)
        
        # For ATM options, vera should be close to zero and may be positive or negative
        # depending on exact parameters, but typically negative
        self.assertLess(vera_atm, 0)
        
        # For OTM options, vera can be positive when d1 is negative enough
        # Let's verify the sign based on d1
        d1, _ = calculate_d1_d2(70, K, T, r, q, vol)
        if d1 < 0:
            self.assertGreater(vera_otm, 0)
        else:
            self.assertLess(vera_otm, 0)

    def test_vera_changes_with_maturity(self):
        # Test how vera changes with different times to maturity
        S, K, r, q, vol = 100, 100, 0.05, 0.02, 0.2
        
        # Calculate vera at different times to maturity
        vera_short = calculate_vera_value(S, K, 0.25, r, q, vol)
        vera_medium = calculate_vera_value(S, K, 1, r, q, vol)
        vera_long = calculate_vera_value(S, K, 2, r, q, vol)
        
        # The magnitude of vera increases with time to maturity
        # since vera includes a factor of T
        self.assertTrue(abs(vera_short) < abs(vera_medium))
        self.assertTrue(abs(vera_medium) < abs(vera_long))
        
        # For ATM options, vera should be negative across different maturities
        # with standard parameters
        self.assertLess(vera_short, 0)
        self.assertLess(vera_medium, 0)
        self.assertLess(vera_long, 0)

    def test_vera_changes_with_vol(self):
        # Test how vera changes with different volatilities
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Calculate vera at different volatility levels
        vera_low_vol = calculate_vera_value(S, K, T, r, q, 0.1)
        vera_mid_vol = calculate_vera_value(S, K, T, r, q, 0.2)
        vera_high_vol = calculate_vera_value(S, K, T, r, q, 0.3)
        
        # The magnitude of vera typically decreases as volatility increases
        # for ATM options with standard parameters
        # Note: This depends on how the factor d1/(vol*sqrt(T)) changes
        # with volatility, which can be complex
        self.assertTrue(abs(vera_low_vol) > abs(vera_mid_vol))
        self.assertTrue(abs(vera_mid_vol) > abs(vera_high_vol))

    def test_vera_with_zero_vol(self):
        # Test case for very low volatility
        S, K, T, r, q = 100, 100, 1, 0.05, 0.02
        
        # Extremely low volatility should be caught by validation
        with self.assertRaises(ValueError):
            calculate_vera_value(S, K, T, r, q, 0)

    def test_vera_with_zero_time(self):
        # Test case for very short time to maturity
        S, K, r, q, vol = 100, 100, 0.05, 0.02, 0.2
        
        # Extremely short time should be caught by validation
        with self.assertRaises(ValueError):
            calculate_vera_value(S, K, 0, r, q, vol)

if __name__ == '__main__':
    unittest.main()
