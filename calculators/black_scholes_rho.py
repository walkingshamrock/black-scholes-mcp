# This file contains the Black-Scholes Rho calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_cdf

def calculate_rho_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Rho of a European option.
    Rho measures the sensitivity of the option price to changes in the interest rate.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put').

    Returns:
        The Rho of the option. By convention, this is the sensitivity for a 1% change in interest rate.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)

    _, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Calculating Rho
    # Rho is typically expressed as the change in option value for a 1% (0.01) change in interest rate
    # Hence, we multiply the result by 0.01 to match market conventions
    
    if option_type == "call":
        # For call options: Rho = K * T * e^(-rT) * N(d2) * 0.01
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) * 0.01
    elif option_type == "put":
        # For put options: Rho = -K * T * e^(-rT) * N(-d2) * 0.01
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) * 0.01
    
    return rho

def register_rho_tool(mcp):
    """Register the Black-Scholes Rho calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Rho",
                "summary": "Calculate option Rho using Black-Scholes model",
                "description": "Returns the Rho of a European call or put option given market parameters. Rho measures the rate of change of the option price with respect to changes in the risk-free interest rate.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_rho(
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
        """Calculate the Black-Scholes Rho for a European call or put option. USE THIS FUNCTION when asked about interest rate
        sensitivity or when specifically asked to calculate rho. Rho measures the rate of change of the option price
        with respect to changes in the risk-free interest rate.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset
            type: Option type, 'call' or 'put'

        Returns:
            A dictionary containing the calculated Rho (for a 1% change in interest rate), input parameters, and model name

        Example:
            calc_black_scholes_rho(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, type="call") ->
            {
              "rho_value": 0.432,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2, "type": "call"},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)
            rho = calculate_rho_value(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type
            )

            return {
                "rho_value": rho,
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
                f"Numerical overflow in Rho calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Rho calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Rho calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_rho
