# This file contains the Black-Scholes Vera calculation logic and tool registration.

import math
from typing import Literal, Dict, Any, cast
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_vera_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Vera (or DrhoDvol) of a European option.
    Vera measures the rate of change of rho with respect to volatility.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put').

    Returns:
        The Vera of the option.
    """
    # Vera is different for calls and puts, but we'll calculate for calls
    # Vera for puts is -K*T*exp(-r*T)*norm_cdf(-d2)*d1/(vol*sqrt(T))
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Vera for call is -K*T*exp(-r*T)*norm_pdf(d2)*d1/(vol*sqrt(T))
    vera = -K * T * math.exp(-r * T) * norm_pdf(d2) * d1 / (vol * math.sqrt(T))
    
    return vera

def register_vera_tool(mcp):
    """Register the Black-Scholes Vera calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Vera",
                "summary": "Calculate option Vera using Black-Scholes model",
                "description": "Returns the Vera of a European call or put option given market parameters. Vera measures the rate of change of rho with respect to volatility.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_vera(S: float, K: float, T: float, r: float, q: float, vol: float, type: Literal["call", "put"]) -> Dict:
        """Calculate the Black-Scholes Vera for a European call or put option. USE THIS FUNCTION when asked about
        the cross-sensitivity between rho and volatility, or when specifically asked to calculate vera.
        Vera measures the rate of change of rho with respect to volatility.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            type: Option type ('call' or 'put')
            
        Returns:
            Vera value
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)
            vera_val = calculate_vera_value(S, K, T, r, q, vol)
            
            # Adjust sign for put options
            if type == "put":
                vera_val = -vera_val
                
            return {
                "vera": vera_val,
                "explanation": f"Vera for {type} option: {vera_val:.6f}. This measures the rate of change of rho with respect to volatility."
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError(f"Error calculating Vera: {str(e)}")
    
    return calc_black_scholes_vera
