import math
from typing import Literal

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("black-scholes")


@mcp.tool()
def calc_black_scholes_price(
    S: float,  # Spot price
    K: float,  # Strike price
    T: float,  # Time to maturity (in years)
    r: float,  # Risk-free rate (annual, decimal)
    q: float,  # Dividend yield (annual, decimal)
    vol: float,  # Volatility (annual, decimal)
    type: Literal["call", "put"],  # Option type
) -> float:
    """Calculate the Black-Scholes price for a European call or put option."""
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
@mcp.tool()
def calc_black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes delta."""
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    if type == "call":
        return math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    elif type == "put":
        return math.exp(-q * T) * (0.5 * (1 + math.erf(d1 / math.sqrt(2))) - 1)
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool()
def calc_black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes vega."""
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return S * math.exp(-q * T) * norm_pdf * math.sqrt(T)


@mcp.tool()
def calc_black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes theta."""
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


@mcp.tool()
def calc_black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes gamma."""
    import math

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return math.exp(-q * T) * norm_pdf / (S * vol * math.sqrt(T))


@mcp.tool()
def calc_black_scholes_rho(S: float, K: float, T: float, r: float, q: float, vol: float, type: Literal["call", "put"]) -> float:
    """Calculate the Black-Scholes rho (sensitivity to interest rate)."""
    import math
    d1 = (math.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if type == "call":
        return K * T * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    elif type == "put":
        return -K * T * math.exp(-r * T) * 0.5 * (1 - math.erf(d2 / math.sqrt(2)))
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool()
def calc_black_scholes_lambda(S: float, K: float, T: float, r: float, q: float, vol: float, type: Literal["call", "put"]) -> float:
    """Calculate the Black-Scholes lambda (elasticity)."""
    price = calc_black_scholes_price(S, K, T, r, q, vol, type)
    delta = calc_black_scholes_delta(S, K, T, r, q, vol, type)
    return (delta * S) / price if price != 0 else float('nan')


@mcp.tool()
def calc_black_scholes_epsilon(S: float, K: float, T: float, r: float, q: float, vol: float, type: Literal["call", "put"]) -> float:
    """Calculate the Black-Scholes epsilon (sensitivity to dividend yield)."""
    import math
    d1 = (math.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
    if type == "call":
        return -T * S * math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    elif type == "put":
        return -T * S * math.exp(-q * T) * 0.5 * (1 - math.erf(d1 / math.sqrt(2)))
    else:
        raise ValueError("type must be 'call' or 'put'")


@mcp.tool()
def calc_black_scholes_charm(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes charm (delta decay)."""
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


@mcp.tool()
def calc_black_scholes_color(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes color (gamma decay)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    color = (
        -math.exp(-q * T)
        * norm_pdf
        / (2 * S * T * vol * math.sqrt(T))
        * (2 * q * T + 1 + d1 * vol * math.sqrt(T))
    )
    return color


@mcp.tool()
def calc_black_scholes_speed(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes speed (rate of change of gamma)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    gamma = math.exp(-q * T) * norm_pdf / (S * vol * math.sqrt(T))
    speed = -gamma / S * (d1 / (vol * math.sqrt(T)) + 1)
    return speed


@mcp.tool()
def calc_black_scholes_ultima(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes ultima (third derivative wrt vol)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vega = S * math.exp(-q * T) * norm_pdf * math.sqrt(T)
    ultima = -vega / (vol**2) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
    return ultima


@mcp.tool()
def calc_black_scholes_vanna(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes vanna (sensitivity of vega to spot)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vanna = -math.exp(-q * T) * norm_pdf * d2 / vol
    return vanna


@mcp.tool()
def calc_black_scholes_vera(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes vera (sensitivity of rho to vol)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vera = K * math.exp(-r * T) * norm_pdf * math.sqrt(T) * d2 / vol
    return vera


@mcp.tool()
def calc_black_scholes_veta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes veta (sensitivity of vega to time)."""
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


@mcp.tool()
def calc_black_scholes_vomma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes vomma (sensitivity of vega to vol)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    vomma = S * math.exp(-q * T) * norm_pdf * math.sqrt(T) * d1 * d2 / vol
    return vomma


@mcp.tool()
def calc_black_scholes_zomma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    type: Literal["call", "put"],
) -> float:
    """Calculate the Black-Scholes zomma (sensitivity of gamma to vol)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    norm_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    zomma = math.exp(-q * T) * norm_pdf * d1 * d2 / (S * vol * math.sqrt(T))
    return zomma
