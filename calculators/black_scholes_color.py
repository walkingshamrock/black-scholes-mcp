# This file contains the Black-Scholes Color calculation logic and tool registration.

import math
from typing import Literal, Dict, Any, cast
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_color_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Color (also known as Gamma decay or DgammaDtime) of a European option.
    Color measures the rate of change of gamma with respect to time.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Color of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Color is same for call and put

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # First, calculate gamma
    gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
    gamma_denominator = S * vol * math.sqrt(T)
    
    if gamma_denominator == 0:
        raise ZeroDivisionError("Cannot calculate Color: vol * sqrt(T) * S is zero")
    
    gamma = gamma_numerator / gamma_denominator
    
    # Then calculate color
    # Color = -gamma * (r - q + d1*vol/(2*sqrt(T)) + (2*q + d1*vol/sqrt(T))/(2*T))
    term1 = r - q + (d1 * vol) / (2 * math.sqrt(T))
    term2 = (2 * q + d1 * vol / math.sqrt(T)) / (2 * T)
    color = -gamma * (term1 + term2)
    
    return color

def register_color_tool(mcp):
    """Register the Black-Scholes Color calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Color",
                "summary": "Calculate option Color using Black-Scholes model",
                "description": "Returns the Color of a European option given market parameters. Color measures the rate of change of gamma with respect to time.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_color(S: float, K: float, T: float, r: float, q: float, vol: float) -> Dict:
        """Calculate the Black-Scholes Color for a European option. USE THIS FUNCTION when asked about
        gamma decay, the rate of change of gamma with respect to time, or when specifically asked to calculate color.
        Color measures how gamma changes as time passes.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            
        Returns:
            Color value
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Color is same for call and put
            color_val = calculate_color_value(S, K, T, r, q, vol)
            return {
                "color": color_val,
                "explanation": f"Color: {color_val:.6f}. This measures the rate of change of gamma with respect to time (gamma decay)."
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError(f"Error calculating Color: {str(e)}")
    
    return calc_black_scholes_color
