# This file contains the Black-Scholes price calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import norm_cdf, calculate_d1_d2, validate_inputs

def calculate_price_value(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    q: float, 
    vol: float, 
    option_type: Literal["call", "put"]
) -> float:
    """
    Calculates the Black-Scholes price for a European call or put option.
    Note: Input validation is expected to be done by the caller using validate_inputs from black_scholes_common.
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, q, vol)

    price: float
    if option_type == "call":
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
    else:
        # This case should ideally be caught by validate_inputs, but as a safeguard:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'.")

    if not math.isfinite(price):
        raise RuntimeError("Calculation resulted in a non-finite price. Please check your input parameters.")
    
    return price

def register_price_tool(mcp):
    """Register the Black-Scholes price calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Price",
                "summary": "Calculate option price using Black-Scholes model",
                "description": "Returns the price of a European call or put option given market parameters.",
                "readOnlyHint": True,  # This tool only calculates and doesn't modify any data
                "idempotentHint": True,  # Same inputs always produce same outputs
            },
        )
    )
    def calc_black_scholes_price(
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
        """Calculate the Black-Scholes price for a European call or put option. USE THIS FUNCTION when asked to calculate
        or determine the theoretical fair value of an option, when pricing European options, or when specifically asked
        to use the Black-Scholes model to calculate option prices.

        The Black-Scholes model is a mathematical formula used in finance to determine the theoretical price
        of European-style options, assuming the asset price follows a geometric Brownian motion with constant drift and volatility.

        Parameters:
            S: Spot price of the underlying asset (must be positive)
            K: Strike price of the option (must be positive)
            T: Time to maturity in years (must be positive)
            r: Risk-free interest rate (annualized, as a decimal)
            q: Dividend yield (annualized, as a decimal)
            vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
            type: Option type, either 'call' or 'put'

        Returns:
            A dictionary containing the calculated option price, input parameters, and model name

        Raises:
            ValueError: When parameters don't meet required constraints (with specific error messages)
            OverflowError: When calculations result in numerical overflow
            ZeroDivisionError: When a divide-by-zero situation occurs
            RuntimeError: For other computational errors

        Example:
            calc_black_scholes_price(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2, type="call") ->
            {
              "option_price": 10.45,
              "input_parameters": {
                "S": 100,
                "K": 100,
                "T": 1,
                "r": 0.05,
                "q": 0.02,
                "vol": 0.20,
                "type": "call"
              },
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            # Validate inputs first
            validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type)

            # Calculate the price using the imported function
            price = calculate_price_value(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type=type
            )

            # Return a structured result with the price, input parameters, and model name
            return {
                "option_price": price,
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
                f"Numerical overflow in calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(f"Division by zero error in calculation: {str(e)}")
        except ValueError as e:  # Catch validation errors specifically
            raise e  # Re-raise validation errors as they are already well-described
        except RuntimeError as e:  # Catch specific runtime errors from calculations
            raise e
        except Exception as e:
            # General catch-all for unexpected errors during calculation
            raise RuntimeError(f"Error in Black-Scholes calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_price