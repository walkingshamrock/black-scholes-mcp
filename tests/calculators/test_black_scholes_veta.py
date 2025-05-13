import unittest
import math
from calculators.black_scholes_veta import calculate_veta_value
from calculators.black_scholes_common import calculate_d1_d2, norm_pdf

class TestBlackScholesVeta(unittest.TestCase):

    def test_calculate_veta_at_the_money(self):
        # Test veta with at-the-money option
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        term1 = q
        term2 = (r - q) * d1 / (vol * math.sqrt(T))
        term3 = (1 + d1 * d2) / (2 * T)
        expected_veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(veta, expected_veta, places=8)

    def test_calculate_veta_in_the_money(self):
        # Test veta with in-the-money option
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        term1 = q
        term2 = (r - q) * d1 / (vol * math.sqrt(T))
        term3 = (1 + d1 * d2) / (2 * T)
        expected_veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(veta, expected_veta, places=8)

    def test_calculate_veta_out_of_the_money(self):
        # Test veta with out-of-the-money option
        S, K, T, r, q, vol = 90, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        term1 = q
        term2 = (r - q) * d1 / (vol * math.sqrt(T))
        term3 = (1 + d1 * d2) / (2 * T)
        expected_veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(veta, expected_veta, places=8)

    def test_veta_short_maturity(self):
        # Test with very short time to maturity (excluding extreme cases that might cause numerical issues)
        S, K, T, r, q, vol = 100, 100, 0.01, 0.05, 0.02, 0.2
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        
        # For short maturity, we just check that the result is finite and not extremely large
        self.assertTrue(math.isfinite(veta))
        # Additionally, Veta should be negative for short maturities (especially at-the-money options)
        self.assertLess(veta, 0)

    def test_veta_long_maturity(self):
        # Test with long time to maturity
        S, K, T, r, q, vol = 100, 100, 5, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        term1 = q
        term2 = (r - q) * d1 / (vol * math.sqrt(T))
        term3 = (1 + d1 * d2) / (2 * T)
        expected_veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(veta, expected_veta, places=8)
    
    def test_veta_high_volatility(self):
        # Test with high volatility
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.5
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        term1 = q
        term2 = (r - q) * d1 / (vol * math.sqrt(T))
        term3 = (1 + d1 * d2) / (2 * T)
        expected_veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
        
        veta = calculate_veta_value(S, K, T, r, q, vol)
        self.assertAlmostEqual(veta, expected_veta, places=8)

if __name__ == '__main__':
    unittest.main()
