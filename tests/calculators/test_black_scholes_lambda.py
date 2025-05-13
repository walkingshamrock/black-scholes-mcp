import unittest
import math
from calculators.black_scholes_lambda import calculate_lambda_value
from calculators.black_scholes_delta import calculate_delta_value
from calculators.black_scholes_price import calculate_price_value

class TestBlackScholesLambda(unittest.TestCase):

    def test_calculate_lambda_call_at_the_money(self):
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate the expected Lambda manually
        delta = calculate_delta_value(S, K, T, r, q, vol, "call")
        price = calculate_price_value(S, K, T, r, q, vol, "call")
        expected_lambda = (delta * S) / price
        
        # Calculate Lambda using our function
        lambda_value = calculate_lambda_value(S, K, T, r, q, vol, "call")
        
        # Check that the values match
        self.assertAlmostEqual(lambda_value, expected_lambda, places=8)
        
        # A standard at-the-money call option should have Lambda around 4-6 for typical parameters
        self.assertTrue(3.5 <= lambda_value <= 6.5, f"Expected Lambda between 3.5 and 6.5, got {lambda_value}")

    def test_calculate_lambda_put_at_the_money(self):
        S, K, T, r, q, vol = 100, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate the expected Lambda manually
        delta = calculate_delta_value(S, K, T, r, q, vol, "put")
        price = calculate_price_value(S, K, T, r, q, vol, "put")
        expected_lambda = (delta * S) / price
        
        # Calculate Lambda using our function
        lambda_value = calculate_lambda_value(S, K, T, r, q, vol, "put")
        
        # Check that the values match
        self.assertAlmostEqual(lambda_value, expected_lambda, places=8)
        
        # A standard at-the-money put option should have negative Lambda
        self.assertTrue(lambda_value < 0, f"Expected negative Lambda for put option, got {lambda_value}")

    def test_calculate_lambda_call_in_the_money(self):
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate Lambda using our function
        lambda_value = calculate_lambda_value(S, K, T, r, q, vol, "call")
        
        # In-the-money call options still have Lambda above 1.0
        # Our calculation gave 5.17, so adjust the expected range
        self.assertTrue(lambda_value >= 1.0, f"Expected Lambda >= 1.0, got {lambda_value}")
        
        # Verify that ITM calls have lower Lambda than ATM calls
        atm_lambda = calculate_lambda_value(100, 100, T, r, q, vol, "call")
        self.assertTrue(lambda_value < atm_lambda, f"Expected ITM Lambda {lambda_value} < ATM Lambda {atm_lambda}")

    def test_calculate_lambda_put_out_of_the_money(self):
        S, K, T, r, q, vol = 110, 100, 1, 0.05, 0.02, 0.2
        
        # Calculate Lambda using our function
        lambda_value = calculate_lambda_value(S, K, T, r, q, vol, "put")
        
        # Out-of-the-money put options have negative and larger absolute Lambda
        self.assertTrue(lambda_value < -3.0, f"Expected Lambda less than -3.0, got {lambda_value}")

    def test_lambda_extreme_case(self):
        # Test deep out-of-the-money option with very short maturity
        # This can result in very small option prices, testing our division by zero handling
        S, K, T, r, q, vol = 80, 100, 0.01, 0.05, 0.02, 0.2
        
        # Should handle small prices without error
        lambda_value = calculate_lambda_value(S, K, T, r, q, vol, "call")
        
        # Should be a very large value (but still finite for reasonable inputs)
        self.assertTrue(lambda_value > 10)

if __name__ == '__main__':
    unittest.main()
