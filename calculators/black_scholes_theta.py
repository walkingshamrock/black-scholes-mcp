# This file contains the Black-Scholes Theta calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf, norm_cdf

def calculate_theta_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Theta of a European option.
    Theta is typically negative for long option positions.
    The value returned is per year. To get per day, divide by 365 (or 252 for trading days).

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option ('call' or 'put').

    Returns:
        The Theta of the option (per year).
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)

    # This term is common for both call and put options
    term1 = -(S * norm_pdf(d1) * vol * math.exp(-q * T)) / (2 * math.sqrt(T))

    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm_cdf(d2)
        term3 = q * S * math.exp(-q * T) * norm_cdf(d1)
        theta = term1 + term2 + term3 # Annualized
    elif option_type == "put":
        term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
        term3 = -q * S * math.exp(-q * T) * norm_cdf(-d1)
        theta = term1 + term2 + term3 # Annualized
    # No else needed as validate_inputs ensures option_type is valid.
    
    return theta

def register_theta_tool(mcp):
    """Register the Black-Scholes Theta calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Theta",
                "summary": "Calculate option Theta using Black-Scholes model",
                "description": "Returns the Theta of a European call or put option given market parameters. Theta measures the rate of change of the option price with respect to the passage of time.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_theta(
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
        """Calculate the Black-Scholes Theta for a European call or put option. USE THIS FUNCTION when asked about time decay,
        time sensitivity of options, or when specifically asked to calculate theta. Theta measures the rate of change of the option price
        with respect to the passage of time (often referred to as time decay).

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset
            type: Option type, 'call' or 'put'

        Returns:
            A dictionary containing the calculated Theta (annualized), input parameters, and model name.
            To get daily theta, divide the returned value by 365 (calendar days) or 252 (trading days).

        Example:
            calc_black_scholes_theta(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, type="call") ->
            {
              "theta_value": -8.21,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2, "type": "call"},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)
            theta = calculate_theta_value(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type
            )

            return {
                "theta_value": theta,
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
                f"Numerical overflow in Theta calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Theta calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Theta calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_theta

    return theta
