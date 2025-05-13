# This file contains the Black-Scholes Ultima calculation logic and tool registration.

import math
from typing import Literal, Dict, Any, cast
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_ultima_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Ultima (or DvommaDvol) of a European option.
    Ultima measures the rate of change of the Vomma with respect to volatility.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put'). Note: Ultima is the same for both calls and puts.

    Returns:
        The Ultima of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)
    
    d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
    
    sqrt_t = math.sqrt(T)
    
    # Ultima = (1/vol) * vomma * (d1*d2 - d1/vol - d2/vol - 1 + 1/(vol*vol))
    # where d2 = d1 - vol*sqrt(T)
    
    d2 = d1 - vol * sqrt_t
    
    # First, we calculate the vomma factor (excluding the common terms)
    vomma_factor = S * math.exp(-q * T) * norm_pdf(d1) * sqrt_t * d1 * d2 / vol
    
    # Then calculate the ultima specific part
    ultima_specific = (d1*d2 - d1/vol - d2/vol - 1 + 1/(vol*vol))
    
    # Ultima
    ultima = (1/vol) * vomma_factor * ultima_specific
    
    return ultima

def register_ultima_tool(mcp):
    """Register the Black-Scholes Ultima calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Ultima",
                "summary": "Calculate option Ultima using Black-Scholes model",
                "description": "Returns the Ultima of a European call or put option given market parameters. Ultima measures the rate of change of the Vomma with respect to volatility.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_ultima(S: float, K: float, T: float, r: float, q: float, vol: float, type: Literal["call", "put"]) -> Dict:
        """Calculate the Black-Scholes Ultima for a European call or put option. USE THIS FUNCTION when asked about
        the third-order sensitivity to volatility changes, or when specifically asked to calculate ultima.
        Ultima measures the rate of change of the Vomma with respect to volatility, which is the third derivative
        of the option price with respect to volatility.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            type: Option type ('call' or 'put')
            
        Returns:
            Ultima value
        """
        try:
            option_type = type
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)
            ultima_val = calculate_ultima_value(S, K, T, r, q, vol, option_type)
            return {
                "ultima": ultima_val,
                "explanation": f"Ultima for {option_type} option: {ultima_val:.6f}. This measures the rate of change of the Vomma with respect to volatility."
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError(f"Error calculating Ultima: {str(e)}")
    
    return calc_black_scholes_ultima
