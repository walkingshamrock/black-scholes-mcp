# This file will contain the Black-Scholes Epsilon calculation logic.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from calculators.black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_epsilon_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Epsilon of a European option (also known as elasticity of vega or DvegaDspot).
    Epsilon measures the percentage change in the option's vega for a percentage change in the underlying price.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Epsilon of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Epsilon is same for call and put

    d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Epsilon is defined as (S/vega) * (∂vega/∂S)
    # The formula for Epsilon is: (d1 * d1 - 1 - d1 / (vol * sqrt(T))) / vol
    
    epsilon = (d1 * d1 - 1 - d1 / (vol * math.sqrt(T))) / vol
    
    return epsilon

def register_epsilon_tool(mcp):
    """Register the Black-Scholes Epsilon calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    
    ToolAnnotations = Any
    
    @mcp.tool(
        annotations={
            "title": "Black-Scholes Option Epsilon",
            "summary": "Calculate option Epsilon using Black-Scholes model",
            "description": "Returns the Epsilon of a European option given market parameters. Epsilon measures the percentage change in the option's vega for a percentage change in the underlying price.",
            "readOnlyHint": True,
            "idempotentHint": True,
        },
    )
    def calc_black_scholes_epsilon(
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
        """Calculate the Black-Scholes Epsilon for a European option. USE THIS FUNCTION when asked about
        the sensitivity of vega to changes in the underlying price, or when specifically asked to calculate epsilon.
        Epsilon measures the percentage change in the option's vega for a percentage change in the underlying price.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            
        Returns:
            Epsilon value
        """
        validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call")  # Epsilon is same for call and put
        epsilon_val = calculate_epsilon_value(S, K, T, r, q, vol)
        return {
            "epsilon": epsilon_val,
            "explanation": f"Epsilon: {epsilon_val:.6f}. This measures the percentage change in the option's vega for a percentage change in the underlying price."
        }
    
    return calc_black_scholes_epsilon
