import math
from typing import Literal, Annotated, cast, Any
from pydantic import Field
# from mcp.server.fastmcp.types import ToolAnnotations
ToolAnnotations = Any

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("black-scholes")


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Option Price",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_price(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes price for a European call or put option.
    
    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        raise ValueError("Invalid input: S, K, T, and vol must be positive.")
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if type == "call":
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(
            d2
        )
    elif type == "put":
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(
            -d1
        )
    else:
        raise ValueError("type must be 'call' or 'put'")
    return price


# --- Black-Scholes Greeks and higher-order Greeks ---
@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Delta",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_delta(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes delta (sensitivity of option price to spot price).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    if type == "call":
        return math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    elif type == "put":
        return math.exp(-q * T) * (0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1)
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Vega",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_vega(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes vega (sensitivity of option price to volatility).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return S * math.exp(-q * T) * norm_pdf * math.sqrt(T)


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Theta",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_theta(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes theta (sensitivity of option price to time decay).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    if type == "call":
        theta = (
            -S * norm_pdf * vol * math.exp(-q * T) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
            + q * S * math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        )
    elif type == "put":
        theta = (
            -S * norm_pdf * vol * math.exp(-q * T) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * 0.5 * (1 - math.erf(d2 / math.sqrt(2)))
            - q * S * math.exp(-q * T) * 0.5 * (1 - math.erf(d1 / math.sqrt(2)))
        )
    else:
        raise ValueError("type must be 'call' or 'put'")
    return theta


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Gamma",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_gamma(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes gamma (sensitivity of delta to spot price).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return math.exp(-q * T) * norm_pdf / (S * vol * math.sqrt(T))


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Rho",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_rho(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes rho (sensitivity of option price to interest rate).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    import math
    
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if type == "call":
        return K * T * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    elif type == "put":
        return -K * T * math.exp(-r * T) * 0.5 * (1 - math.erf(d2 / math.sqrt(2)))
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Lambda",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_lambda(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes lambda (elasticity: percent change in option price per percent change in spot price).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    price = calc_black_scholes_price(S, K, T, r, q, vol, type)
    delta = calc_black_scholes_delta(S, K, T, r, q, vol, type)
    return (delta * S) / price if price != 0 else float('nan')


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Epsilon",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_epsilon(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes epsilon (sensitivity of option price to dividend yield).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    import math
    
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    if type == "call":
        return -T * S * math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    elif type == "put":
        return -T * S * math.exp(-q * T) * 0.5 * (1 - math.erf(d1 / math.sqrt(2)))
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Charm",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_charm(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes charm (delta decay: sensitivity of delta to time).

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put'
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    charm = (
        -math.exp(-q * T)
        * norm_pdf
        * (2 * (r - q) * T - d2 * vol * math.sqrt(T))
        / (2 * T * vol * math.sqrt(T))
    )
    if type == "call":
        return charm + q * math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    elif type == "put":
        return charm - q * math.exp(-q * T) * 0.5 * (1 - math.erf(d1 / math.sqrt(2)))
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Color",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_color(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes color (gamma decay: sensitivity of gamma to time).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    color = (
        -math.exp(-q * T)
        * norm_pdf
        / (2 * S * T * vol * math.sqrt(T))
        * (2 * q * T + 1 + d1 * vol * math.sqrt(T))
    )
    return color


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Speed",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_speed(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes speed (rate of change of gamma with respect to spot price).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    gamma = math.exp(-q * T) * norm_pdf / (S * vol * math.sqrt(T))
    speed = -gamma / S * (d1 / (vol * math.sqrt(T)) + 1)
    return speed


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Ultima",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_ultima(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes ultima (third derivative of option price with respect to volatility).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vega = S * math.exp(-q * T) * norm_pdf * math.sqrt(T)
    ultima = -vega / (vol**2) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
    return ultima


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Vanna",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_vanna(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes vanna (sensitivity of vega to spot price).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vanna = -math.exp(-q * T) * norm_pdf * d2 / vol
    return vanna


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Vera",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_vera(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes vera (sensitivity of rho to volatility).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vera = K * math.exp(-r * T) * norm_pdf * math.sqrt(T) * d2 / vol
    return vera


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Veta",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_veta(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes veta (sensitivity of vega to time to maturity).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    veta = (
        S
        * math.exp(-q * T)
        * norm_pdf
        * math.sqrt(T)
        * (q + (d1 * vol) / (2 * math.sqrt(T)))
    )
    return veta


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Vomma",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_vomma(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes vomma (sensitivity of vega to volatility).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vomma = S * math.exp(-q * T) * norm_pdf * math.sqrt(T) * d1 * d2 / vol
    return vomma


@mcp.tool(
    annotations=cast(ToolAnnotations, {
        "title": "Calculate Black-Scholes Zomma",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    })
)
def calc_black_scholes_zomma(
    S: Annotated[float, Field(description="Spot price of the underlying asset", gt=0)],
    K: Annotated[float, Field(description="Strike price of the option", gt=0)],
    T: Annotated[float, Field(description="Time to maturity in years", gt=0)],
    r: Annotated[float, Field(description="Risk-free interest rate (annualized, as a decimal)")],
    q: Annotated[float, Field(description="Dividend yield (annualized, as a decimal)")],
    vol: Annotated[float, Field(description="Volatility of the underlying asset (annualized, as a decimal)", gt=0)],
    type: Annotated[Literal["call", "put"], Field(description="Option type: 'call' or 'put'")],
) -> float:
    """Calculate the Black-Scholes zomma (sensitivity of gamma to volatility).

    Note: The 'type' argument is included for interface consistency but is not used in the calculation.

    Parameters:
        S: Spot price of the underlying asset (must be positive)
        K: Strike price of the option (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized, as a decimal)
        q: Dividend yield (annualized, as a decimal)
        vol: Volatility of the underlying asset (annualized, as a decimal, must be positive)
        type: Option type, either 'call' or 'put' (not used)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    zomma = math.exp(-q * T) * norm_pdf * d1 * d2 / (S * vol * math.sqrt(T))
    return zomma
