# This file contains the Black-Scholes Speed calculation logic and tool registration.

import math
from typing import Literal, Dict, Any, cast
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_speed_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Speed of a European option.
    Speed is the third derivative of option price with respect to underlying price.
    It is also known as DgammaDspot or the rate of change of Gamma with respect to spot price.
    Speed is the same for call and put options.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Speed of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Speed is same for call and put

    d1, _ = calculate_d1_d2(S, K, T, r, q, vol)

    # Speed = -gamma * (1 + d1/(S * vol * sqrt(T))) / S
    # First calculate gamma
    gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
    gamma_denominator = S * vol * math.sqrt(T)
    
    if gamma_denominator == 0:
        # This case should ideally be avoided by validate_inputs (T>0, vol>0, S>0)
        return float('inf')  # Or handle as an error, though inputs are validated
    
    gamma = gamma_numerator / gamma_denominator
    
    # Now calculate speed
    factor = 1 + d1 / (S * vol * math.sqrt(T))
    speed = -gamma * factor / S
    
    return speed

def register_speed_tool(mcp):
    """Register the Black-Scholes Speed calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Speed",
                "summary": "Calculate option Speed using Black-Scholes model",
                "description": "Returns the Speed of a European call or put option given market parameters. Speed is the third derivative of option price with respect to underlying price, or the rate of change of Gamma with respect to spot price.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_speed(S: float, K: float, T: float, r: float, q: float, vol: float) -> Dict:
        """Calculate the Black-Scholes Speed for a European option. USE THIS FUNCTION when asked about
        the third derivative of option price with respect to underlying price, or when specifically asked
        to calculate speed. Speed is the rate of change of Gamma with respect to spot price.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            
        Returns:
            Speed value
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Speed is same for call and put
            speed_val = calculate_speed_value(S, K, T, r, q, vol)
            return {
                "speed": speed_val,
                "explanation": f"Speed: {speed_val:.6f}. This measures the rate of change of Gamma with respect to the underlying price."
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError(f"Error calculating Speed: {str(e)}")
    
    return calc_black_scholes_speed
