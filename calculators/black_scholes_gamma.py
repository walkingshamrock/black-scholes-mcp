# This file contains the Black-Scholes Gamma calculation logic and tool registration.

import math
from typing import Literal, Annotated, cast, Any, Dict
from pydantic import Field

ToolAnnotations = Any

from .black_scholes_common import validate_inputs, calculate_d1_d2, norm_pdf

def calculate_gamma_value(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """
    Calculates the Gamma of a European option.
    Gamma is the same for call and put options.

    Args:
        S: Spot price of the underlying asset.
        K: Strike price of the option.
        T: Time to maturity (in years).
        r: Risk-free interest rate (annual, decimal).
        q: Annual dividend yield of the underlying asset (decimal).
        vol: Volatility of the underlying asset (annual, decimal).

    Returns:
        The Gamma of the option.
    """
    validate_inputs(S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call") # Gamma is same for call and put

    d1, _ = calculate_d1_d2(S, K, T, r, q, vol)

    # Gamma = exp(-qT) * N'(d1) / (S * vol * sqrt(T))
    numerator = math.exp(-q * T) * norm_pdf(d1)
    denominator = S * vol * math.sqrt(T)
    
    if denominator == 0:
        # This case should ideally be avoided by validate_inputs (T>0, vol>0, S>0)
        # However, if it occurs, Gamma would be infinite or undefined.
        # Depending on context, could return float('inf') or raise error.
        # For now, let's assume validate_inputs prevents this.
        # If S, vol, or T are zero, d1 might also be problematic.
        return float('inf') # Or handle as an error, though inputs are validated

    gamma = numerator / denominator
    return gamma

def register_gamma_tool(mcp):
    """Register the Black-Scholes Gamma calculation tool with the MCP server.
    
    Args:
        mcp: The FastMCP instance to register the tool with.
        
    Returns:
        The registered tool function.
    """
    @mcp.tool(
        annotations=cast(
            ToolAnnotations,
            {
                "title": "Black-Scholes Option Gamma",
                "summary": "Calculate option Gamma using Black-Scholes model",
                "description": "Returns the Gamma of a European call or put option given market parameters. Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price.",
                "readOnlyHint": True,
                "idempotentHint": True,
            },
        )
    )
    def calc_black_scholes_gamma(
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
        """Calculate the Black-Scholes Gamma for a European option. USE THIS FUNCTION when asked about the rate of change
        of delta, second-order price sensitivity, convexity of options, or when specifically asked to calculate gamma.
        Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price and is crucial
        for understanding how delta hedges need to be adjusted as markets move.

        Gamma is the same for both call and put options with the same parameters.

        Parameters:
            S: Spot price of the underlying asset
            K: Strike price of the option
            T: Time to maturity in years
            r: Risk-free interest rate
            q: Dividend yield
            vol: Volatility of the underlying asset

        Returns:
            A dictionary containing the calculated Gamma, input parameters, and model name

        Example:
            calc_black_scholes_gamma(S=100, K=100, T=1, r=0.05, q=0.02, vol=0.2) ->
            {
              "gamma_value": 0.0189,
              "input_parameters": {"S": 100, "K": 100, "T": 1, "r": 0.05, "q": 0.02, "vol": 0.2},
              "model": "black-scholes",
              "status": "success"
            }
        """
        try:
            validate_inputs(
                S=S, K=K, T=T, r=r, q=q, vol=vol, option_type="call"
            )  # Gamma is the same for call and put
            
            gamma = calculate_gamma_value(S=S, K=K, T=T, r=r, q=q, vol=vol)

            return {
                "gamma_value": gamma,
                "input_parameters": {"S": S, "K": K, "T": T, "r": r, "q": q, "vol": vol},
                "model": "black-scholes",
                "status": "success",
            }
        except OverflowError as e:
            raise OverflowError(
                f"Numerical overflow in Gamma calculation. Try using more moderate input values: {str(e)}"
            )
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"Division by zero error in Gamma calculation: {str(e)}"
            )
        except ValueError as e:
            raise e
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Error in Black-Scholes Gamma calculation: {str(e)}")
    
    # Return the registered function for potential further use
    return calc_black_scholes_gamma
