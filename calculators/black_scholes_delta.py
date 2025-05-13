# This file contains the Black-Scholes Delta calculation logic and tool registration.
import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import norm_cdf, calculate_d1_d2, validate_inputs

def calculate_delta_value(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    q: float, 
    vol: float, 
    option_type: Literal["call", "put"]
) -> float:
    """
    Calculates the Black-Scholes Delta for a European call or put option.
    Note: Input validation is expected to be done by the caller.
    """
    d1, _ = calculate_d1_d2(S, K, T, r, q, vol) # d2 is not needed for delta

    delta: float
    if option_type == "call":
        delta = math.exp(-q * T) * norm_cdf(d1)
    elif option_type == "put":
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1)
    else:
        # This case should ideally be caught by validate_inputs, but as a safeguard:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'.")

    if not math.isfinite(delta):
        raise RuntimeError("Delta calculation resulted in a non-finite value. Please check your input parameters.")
    
    return delta

def register_delta_tool(mcp):
    """Register the Black-Scholes Delta calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Delta",
                "summary": "Calculate option Delta using Black-Scholes model",
                "description": "Returns the Delta of a European call or put option given market parameters. Delta measures the rate of change of the option price with respect to changes in the underlying asset's price.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_delta(
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
        """Calculate the Black-Scholes Delta for a European call or put option. USE THIS FUNCTION when asked about option hedging,
        directional exposure, or when specifically asked to calculate delta. Delta measures the rate of change of the option price
        with respect to changes in the underlying asset's price and is often interpreted as the equivalent exposure to the underlying asset.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset
            type: Option type, 'call' or 'put'

        Returns:
            A dictionary containing the calculated Delta, input parameters, and model name

        Example:
            calc_black_scholes_delta(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, type="call") ->
            {
              "delta_value": 0.6368,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2, "type": "call"},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)
            delta = calculate_delta_value(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type
            )

            return {
                "delta_value": delta,
                "input_parameters": {
                    "S": S,
                    "K": K,
                    "T": T,
                    "r": r,
                    "q": q,
                    "vol": vol,
                    "type": type,
                },
                "model": "black-scholes",
                "status": "success",
            }
        except OverflowError as e:
            raise OverflowError(
                f"Numerical overflow in Delta calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Delta calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Delta calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_delta
