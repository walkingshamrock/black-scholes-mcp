# This file contains the Black-Scholes Zomma calculation logic and tool registration.

import math
from typing import Literal, Dict, Any, cast
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_zomma_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Zomma of a European option.
    Zomma measures the rate of change of gamma with respect to changes in volatility.
    It is also known as DgammaDvol.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Zomma of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Zomma is same for call and put

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Calculate zomma:
    # Zomma = (d1 * d2 - 1) * gamma / vol
    # First, calculate gamma
    gamma_numerator = math.exp(-q * T) * norm_pdf(d1)
    gamma_denominator = S * vol * math.sqrt(T)
    
    if gamma_denominator == 0:
        # This case should ideally be avoided by validate_inputs (T>0, vol>0, S>0)
        raise ZeroDivisionError("Cannot calculate Zomma: vol * sqrt(T) * S is zero")
    
    gamma = gamma_numerator / gamma_denominator
    
    # Then calculate Zomma
    zomma = (d1 * d2 - 1) * gamma / vol
    
    return zomma

def register_zomma_tool(mcp):
    """Register the Black-Scholes Zomma calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Zomma",
                "summary": "Calculate option Zomma using Black-Scholes model",
                "description": "Returns the Zomma of a European option given market parameters. Zomma measures the rate of change of gamma with respect to changes in volatility.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_zomma(S: float, K: float, T: float, r: float, q: float, vol: float) -> Dict:
        """Calculate the Black-Scholes Zomma for a European option. USE THIS FUNCTION when asked about
        the sensitivity of gamma to changes in volatility, or when specifically asked to calculate zomma.
        Zomma measures the rate of change of gamma with respect to changes in volatility.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            
        Returns:
            Zomma value
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Zomma is same for call and put
            zomma_val = calculate_zomma_value(S, K, T, r, q, vol)
            return {
                "zomma": zomma_val,
                "explanation": f"Zomma: {zomma_val:.6f}. This measures the rate of change of gamma with respect to changes in volatility."
            }
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise RuntimeError(f"Error calculating Zomma: {str(e)}")
    
    return calc_black_scholes_zomma
