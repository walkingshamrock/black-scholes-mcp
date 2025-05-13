# This file contains the Black-Scholes Veta calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_veta_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Veta (also known as DvegaDtime) of a European option.
    Veta measures the rate of change of the option's vega with respect to the passage of time.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Veta of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Veta is same for call and put

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Veta calculation formula
    # Veta = -S * e^(-q*T) * N'(d1) * sqrt(T) * (q + ((r-q)*d1)/(vol*sqrt(T)) + ((1+d1*d2)/(2*T)))
    
    term1 = q
    term2 = (r - q) * d1 / (vol * math.sqrt(T))
    term3 = (1 + d1 * d2) / (2 * T)
    
    veta = -S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) * (term1 + term2 + term3)
    
    return veta

def register_veta_tool(mcp):
    """Register the Black-Scholes Veta calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Veta",
                "summary": "Calculate option Veta using Black-Scholes model",
                "description": "Returns the Veta of a European option given market parameters. Veta measures the rate of change of the option's vega with respect to the passage of time.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_veta(
        S: Annotated[
            float,
            Field(description="Spot price of the underlying asset", gt=0.0, example=100.0),
        ],
        K: Annotated[
            float, Field(description="Strike price of the option", gt=0.0, example=100.0)
        ],
        T: Annotated[
            float, Field(description="Time to maturity in years", gt=0.0, example=1.0)
        ],
        r: Annotated[
            float,
            Field(
                description="Risk-free interest rate (annualized, as a decimal)",
                example=0.05,
            ),
        ],
        q: Annotated[
            float,
            Field(description="Dividend yield (annualized, as a decimal)", example=0.02),
        ],
        vol: Annotated[
            float,
            Field(
                description="Volatility of the underlying asset (annualized, as a decimal)",
                gt=0.0,
                example=0.2,
            ),
        ],
    ) -> Dict:
        """Calculate the Black-Scholes Veta for a European option. USE THIS FUNCTION when asked about
        how vega changes with time, or when specifically asked to calculate veta.
        Veta measures the rate of change of the option's vega with respect to the passage of time.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            
        Returns:
            Veta value
        """
        validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Veta is same for call and put
        veta_val = calculate_veta_value(S, K, T, r, q, vol)
        return {
            "veta": veta_val,
            "explanation": f"Veta: {veta_val:.6f}. This measures the rate of change of the option's vega with respect to the passage of time."
        }
    
    return calc_black_scholes_veta
