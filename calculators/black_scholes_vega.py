# This file contains the Black-Scholes Vega calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_vega_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Vega of a European option.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Vega of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call") # Vega is same for call and put

    d1, _ = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Vega = S * exp(-qT) * N'(d1) * sqrt(T)
    # N'(d1) is the PDF of the standard normal distribution at d1.
    # We return the raw Vega. Some contexts expect Vega per 1% change in volatility (i.e., divided by 100).
    
    vega = S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)
    return vega

def register_vega_tool(mcp):
    """Register the Black-Scholes Vega calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Vega",
                "summary": "Calculate option Vega using Black-Scholes model",
                "description": "Returns the Vega of a European option given market parameters. Vega measures the rate of change of the option price with respect to changes in the underlying asset's implied volatility.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_vega(
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
        """Calculate the Black-Scholes Vega for a European option. USE THIS FUNCTION when asked about volatility sensitivity,
        volatility exposure, or when specifically asked to calculate vega. Vega measures the rate of change of the option price
        with respect to changes in the underlying asset's implied volatility.

        Vega is the same for both call and put options with the same parameters.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset

        Returns:
            A dictionary containing the calculated Vega, input parameters, and model name

        Example:
            calc_black_scholes_vega(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2) ->
            {
              "vega_value": 39.89,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call"
            )  # Vega is the same for call and put
            
            vega = calculate_vega_value(S=S, K=K, T=T, r=r, q=q, vol=vol)

            return {
                "vega_value": vega,
                "input_parameters": {"S": S, "K": K, "T": T, "r": r, "q": q, "vol": vol},
                "model": "black-scholes",
                "status": "success",
            }
        except OverflowError as e:
            raise OverflowError(
                f"Numerical overflow in Vega calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Vega calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Vega calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_vega
