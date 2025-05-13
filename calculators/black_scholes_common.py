# This file will contain common functions and utilities for Black-Scholes calculations.

import math
from typing import Literal

def norm_cdf(x: float) -> float:
    """Calculate the cumulative distribution function for a standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x: float) -> float:
    """Probability density function of the standard normal distribution."""
    return (1.0 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * x * x)

def validate_inputs(
    S: float, K: float, T: float, r: float, q: float, vol: float, option_type: Literal["call", "put"]
) -> None:
    """
    Validate the input parameters for Black-Scholes calculations.
    Raises ValueError if any input is invalid.
    """
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        # General check first, then more specific errors can be caught if needed by individual checks below
        pass # Covered by more specific checks below, but good for a quick top-level check

    # Specific validation with detailed error messages
    if S <= 0:
        raise ValueError(f"Spot price (S) must be positive. Got: {S}")
    if K <= 0:
        raise ValueError(f"Strike price (K) must be positive. Got: {K}")
    if T <= 0:
        raise ValueError(f"Time to maturity (T) must be positive. Got: {T}")
    if vol <= 0:
        raise ValueError(f"Volatility (vol) must be positive. Got: {vol}")

    # Check for extreme values that might cause computational issues
    if S > 1e12: # Increased limit slightly from 1e10 to allow for larger nominals if necessary
        raise ValueError(f"Spot price (S) is too large: {S}. Consider scaling down your inputs.")
    if K > 1e12: # Increased limit slightly
        raise ValueError(f"Strike price (K) is too large: {K}. Consider scaling down your inputs.")
    if T > 100:
        raise ValueError(f"Time to maturity (T) is unusually large: {T} years. Please verify this value.")
    if vol > 5: # Reduced limit from 10, as >500% vol is extremely rare and likely an error
        raise ValueError(f"Volatility (vol) value {vol} seems unusually high. Volatility is typically expressed as a decimal (e.g., 0.2 for 20%).")
    if abs(r) > 1:
        raise ValueError(f"Risk-free rate (r) value {r} seems unusual. Please confirm r is expressed as a decimal (e.g., 0.05 for 5%).")
    if q < 0:
        raise ValueError(f"Dividend yield (q) cannot be negative. Got: {q}")
    if q > 1: # Dividend yield > 100% is highly unusual
        raise ValueError(f"Dividend yield (q) value {q} seems unusually high. Please confirm q is expressed as a decimal (e.g., 0.02 for 2%).")

    if option_type not in ["call", "put"]:
        raise ValueError(f"Option type must be 'call' or 'put'. Got: {option_type}")

def calculate_d1_d2(S: float, K: float, T: float, r: float, q: float, vol: float) -> tuple[float, float]:
    """
    Calculate d1 and d2, parameters used in the Black-Scholes formula.
    """
    # Ensure T and vol are positive to avoid math errors (already validated by validate_inputs, but good for safety)
    if T <= 0 or vol <= 0:
        # This case should ideally be caught by validate_inputs before reaching here.
        # However, if called directly, this prevents division by zero or log of non-positive.
        raise ValueError("Time to maturity (T) and volatility (vol) must be positive for d1/d2 calculation.")

    denominator = vol * math.sqrt(T)
    if denominator == 0:
        # This can happen if T or vol is extremely small, effectively zero, or exactly zero.
        # Handle by raising an error, as d1/d2 would be undefined.
        # Consider what d1/d2 should be in edge cases (e.g. T->0 or vol->0)
        # For T->0, d1 tends to +/- infinity depending on S/K.
        # For vol->0, d1 tends to +/- infinity.
        # A practical approach is to disallow T=0 or vol=0 via validate_inputs.
        raise ZeroDivisionError("Cannot calculate d1/d2: vol * sqrt(T) is zero. Ensure T and vol are positive and non-zero.")

    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / denominator
    d2 = d1 - denominator
    return d1, d2
