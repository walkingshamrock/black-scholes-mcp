# This file will contain the Black-Scholes Vomma calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_vomma_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Vomma (or Volga) of a European option.
    Vomma measures the rate of change of vega with respect to volatility.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put'). Note: Vomma is the same for both calls and puts.

    Returns:
        The Vomma of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)
    
    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    sqrt_t = math.sqrt(T)
    
    # Vomma = S * exp(-q*T) * norm_pdf(d1) * sqrt(T) * (d1*d2) / vol
    vomma = S * math.exp(-q * T) * norm_pdf(d1) * sqrt_t * (d1 * d2) / vol
    
    return vomma

def register_vomma_tool(mcp):
    """Register the Black-Scholes Vomma calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Vomma",
                "summary": "Calculate option Vomma using Black-Scholes model",
                "description": "Returns the Vomma of a European call or put option given market parameters. Vomma measures the rate of change of vega with respect to volatility.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_vomma(
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
        type: Annotated[
            Literal["call", "put"],
            Field(description="Option type: 'call' or 'put'", example="call"),
        ],
    ) -> Dict:
        """Calculate the Black-Scholes Vomma for a European call or put option. USE THIS FUNCTION when asked about
        the second-order sensitivity to volatility changes, or when specifically asked to calculate vomma.
        Vomma measures the rate of change of vega with respect to volatility, which is the second derivative
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
            Vomma value
        """
        option_type = type
        validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)
        vomma_val = calculate_vomma_value(S, K, T, r, q, vol, option_type)
        return {
            "vomma": vomma_val,
            "explanation": f"Vomma for {option_type} option: {vomma_val:.6f}. This measures the rate of change of vega with respect to volatility."
        }
    
    return calc_black_scholes_vomma
