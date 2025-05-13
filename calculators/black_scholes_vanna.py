# This file contains the Black-Scholes Vanna calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_vanna_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Vanna (also known as DdeltaDvol) of a European option.
    Vanna measures the rate of change of delta with respect to changes in volatility,
    or equivalently, the rate of change of vega with respect to changes in the underlying price.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Vanna of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call") # Vanna is same for call and put

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Vanna = -e^(-qt) * N'(d1) * d2 / vol
    # Where N'(d1) is the standard normal probability density function at d1
    
    vanna = -math.exp(-q * T) * norm_pdf(d1) * d2 / vol
    
    return vanna

def register_vanna_tool(mcp):
    """Register the Black-Scholes Vanna calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Vanna",
                "summary": "Calculate option Vanna using Black-Scholes model",
                "description": "Returns the Vanna of a European option given market parameters. Vanna measures the rate of change of delta with respect to changes in volatility, or the sensitivity of vega to changes in the underlying price.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_vanna(
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
        """Calculate the Black-Scholes Vanna for a European option. USE THIS FUNCTION when asked about the cross sensitivity 
        between delta and volatility, or vega and spot price, or when specifically asked to calculate vanna. 
        Vanna (also known as DdeltaDvol or DvegaDspot) measures the rate of change of delta with respect to changes in volatility,
        or equivalently, the rate of change of vega with respect to changes in the underlying price.

        Vanna is the same for both call and put options with identical parameters.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset

        Returns:
            A dictionary containing the calculated Vanna, input parameters, and model name

        Example:
            calc_black_scholes_vanna(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2) ->
            {
              "vanna_value": -0.0948,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call"
            )  # Vanna is the same for call and put
            
            vanna = calculate_vanna_value(S=S, K=K, T=T, r=r, q=q, vol=vol)

            return {
                "vanna_value": vanna,
                "input_parameters": {"S": S, "K": K, "T": T, "r": r, "q": q, "vol": vol},
                "model": "black-scholes",
                "status": "success",
            }
        except OverflowError as e:
            raise OverflowError(
                f"Numerical overflow in Vanna calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Vanna calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Vanna calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_vanna
