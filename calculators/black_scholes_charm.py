# This file contains the Black-Scholes Charm calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf, norm_cdf

def calculate_charm_value(S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]) -> float:
    """
    Calculates the Charm (also known as Delta Decay or DdeltaDtime) of a European option.
    Charm measures the instantaneous rate of change of delta with respect to time.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).
        option_type: Type of the option - "call" or "put".

    Returns:
        The Charm of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=option_type)

    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)
    
    # Calculate common components
    common_term = -math.exp(-q * T) * norm_pdf(d1) / (2 * T)
    
    # The formula for Charm is different for calls and puts
    if option_type == "call":
        # For calls: Charm = -e^(-qT) * [N'(d1) * (2(r-q)/(vol*sqrt(T)) - d2/(2*T)) + q*N(d1)]
        charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) - q * math.exp(-q * T) * norm_cdf(d1)
    else:  # put
        # For puts: Charm = -e^(-qT) * [N'(d1) * (2(r-q)/(vol*sqrt(T)) - d2/(2*T)) - q*N(-d1)]
        charm = common_term * (2 * (r - q) / (vol * math.sqrt(T)) - d2 / (2 * T)) + q * math.exp(-q * T) * norm_cdf(-d1)

    return charm

def register_charm_tool(mcp):
    """Register the Black-Scholes Charm calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Charm",
                "summary": "Calculate option Charm using Black-Scholes model",
                "description": "Returns the Charm (Delta Decay) of a European call or put option given market parameters. Charm measures the instantaneous rate of change of delta with respect to time.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_charm(
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
        """Calculate the Black-Scholes Charm for a European call or put option. USE THIS FUNCTION when asked about
        delta decay, the change in delta over time, or when specifically asked to calculate charm.
        Charm (also known as Delta Decay or DdeltaDtime) measures the instantaneous rate of change of delta with respect to time.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset
            type: Option type, 'call' or 'put'

        Returns:
            A dictionary containing the calculated Charm, input parameters, and model name

        Example:
            calc_black_scholes_charm(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, type="call") ->
            {
              "charm_value": -0.0635,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2, "type": "call"},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)
            charm = calculate_charm_value(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type
            )

            return {
                "charm_value": charm,
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
                f"Numerical overflow in Charm calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Charm calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Charm calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_charm
