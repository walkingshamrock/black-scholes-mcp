import unittest
import math
from calculators.black_scholes_charm import calculate_charm_value
from calculators.black_scholes_common import calculate_d1_d2, norm_pdf, norm_cdf

class TestBlackScholesCharm(unittest.TestCase):

    def test_calculate_charm_call_at_the_money(self):
        # Test charm with at-the-money call option
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        common_term = -math.exp(-q * T) * norm_pdf(d1) / (2 * T)
        expected_charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) - q * math.exp(-q * T) * norm_cdf(d1)
        
        charm = calculate_charm_value(S, K, T, r, q, vol, "call")
        self.assertAlmostEqual(charm, expected_charm, places=8)

    def test_calculate_charm_put_at_the_money(self):
        # Test charm with at-the-money put option
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        common_term = -math.exp(-q * T) * norm_pdf(d1) / (2 * T)
        expected_charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) + q * math.exp(-q * T) * norm_cdf(-d1)
        
        charm = calculate_charm_value(S, K, T, r, q, vol, "put")
        self.assertAlmostEqual(charm, expected_charm, places=8)

    def test_calculate_charm_call_in_the_money(self):
        # Test charm with in-the-money call option
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        common_term = -math.exp(-q * T) * norm_pdf(d1) / (2 * T)
        expected_charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) - q * math.exp(-q * T) * norm_cdf(d1)
        
        charm = calculate_charm_value(S, K, T, r, q, vol, "call")
        self.assertAlmostEqual(charm, expected_charm, places=8)

    def test_calculate_charm_put_out_of_the_money(self):
        # Test charm with out-of-the-money put option
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
        
        # Calculate expected value
        common_term = -math.exp(-q * T) * norm_pdf(d1) / (2 * T)
        expected_charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) + q * math.exp(-q * T) * norm_cdf(-d1)
        
        charm = calculate_charm_value(S, K, T, r, q, vol, "put")
        self.assertAlmostEqual(charm, expected_charm, places=8)

    def test_charm_short_maturity(self):
        # Test with very short time to maturity
        S, K, T, r, q, vol = 100, 100, 0.01, 0.05, 0.02, 0.2
        
        # For call option
        charm_call = calculate_charm_value(S, K, T, r, q, vol, "call")
        self.assertTrue(math.isfinite(charm_call), f"Charm calculation for short maturity call resulted in {charm_call}")
        
        # For put option
        charm_put = calculate_charm_value(S, K, T, r, q, vol, "put")
        self.assertTrue(math.isfinite(charm_put), f"Charm calculation for short maturity put resulted in {charm_put}")

    def test_charm_long_maturity(self):
        # Test with long time to maturity
        S, K, T, r, q, vol = 100, 100, 5, 0.05, 0.02, 0.2
        
        # For call option
        charm_call = calculate_charm_value(S, K, T, r, q, vol, "call")
        self.assertTrue(math.isfinite(charm_call), f"Charm calculation for long maturity call resulted in {charm_call}")
        
        # For put option
        charm_put = calculate_charm_value(S, K, T, r, q, vol, "put")
        self.assertTrue(math.isfinite(charm_put), f"Charm calculation for long maturity put resulted in {charm_put}")
    
    def test_put_call_parity_relation(self):
        # Test that put-call parity relation holds for Charm:
        # Charm_call - Charm_put = q * e^(-qT) 
        # (with a small difference due to floating point precision)
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        charm_call = calculate_charm_value(S, K, T, r, q, vol, "call")
        charm_put = calculate_charm_value(S, K, T, r, q, vol, "put")
        
        # The difference should be approximately q * e^(-qT)
        expected_diff = q * math.exp(-q * T)
        actual_diff = charm_call - charm_put
        
        self.assertAlmostEqual(actual_diff, -expected_diff, places=8)

if __name__ == '__main__':
    unittest.main()
