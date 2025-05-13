# This file will contain the Black-Scholes Lambda calculation logic.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from calculators.black_scholes_common import validate_inputs
from calculators.black_scholes_delta import calculate_delta_value
from calculators.black_scholes_price import calculate_price_value

def calculate_lambda_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Lambda (Elasticity or Omega) of a European option.
    Lambda measures the percentage change in option value per percentage change in the underlying price.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put').

    Returns:
        The Lambda (Elasticity) of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)
    
    # Calculate delta first
    delta = calculate_delta_value(S, K, T, r, q, vol, option_type)
    
    # Calculate option price
    price = calculate_price_value(S, K, T, r, q, vol, option_type)
    
    # Avoid division by zero
    if abs(price) < 1e-10:
        # For very small prices, Lambda would approach infinity
        # Return a large value with appropriate sign instead
        return math.copysign(float('inf'), delta)
    
    # Lambda = (Delta * S) / Option Price
    lambda_value = (delta * S) / price
    
    return lambda_value

def register_lambda_tool(mcp):
    """Register the Black-Scholes Lambda calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Lambda (Elasticity)",
                "summary": "Calculate option Lambda (Elasticity) using Black-Scholes model",
                "description": "Returns the Lambda (Elasticity) of a European call or put option given market parameters. Lambda measures the percentage change in option value per percentage change in the underlying price.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_lambda(
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
        """Calculate the Black-Scholes Lambda (Elasticity) for a European call or put option. USE THIS FUNCTION when asked about
        option leverage, elasticity, or when specifically asked to calculate lambda. Lambda measures the percentage change in option value 
        per percentage change in the underlying price (also known as elasticity or omega).

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate (annualized, decimal)
            q: Dividend yield (annualized, decimal)
            vol: Volatility of the underlying asset (annualized, decimal)
            type: Option type ('call' or 'put')
            
        Returns:
            Lambda value (unit-less ratio)
        """
        option_type = type
        validate_inputs(S, K, T, r, q, vol, option_type)
        lambda_val = calculate_lambda_value(S, K, T, r, q, vol, option_type)
        return {
            "lambda": lambda_val,
            "explanation": f"Lambda (Elasticity) for {option_type} option: {lambda_val:.6f}. This means the option's value changes by {abs(lambda_val):.2f}% for a 1% change in the underlying asset's price."
        }
    
    return calc_black_scholes_lambda
