"""Example tools for weather and math operations used by React agents."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from langchain.tools import tool

from src.utils import Logger


@tool
def weather_lookup_tool(city: str, unit: str = "c") -> str:
    """Return mock current-weather text for a supported city.

    Args:
        city: City name used for lookup. Best-effort matching expects lowercase,
            ASCII-friendly input such as `istanbul`, `ankara`, `new york`.
            Avoid country suffixes (use `ankara`, not `ankara, turkey`).
        unit: Temperature unit. Use `c` for Celsius (default) or `f` for Fahrenheit.

    Returns:
        A short human-readable weather sentence that includes city, timestamp,
        temperature, and condition; or a clear "not found" message when the
        city does not exist in the mock dataset.
    """
    Logger.info("weather_lookup_tool called | city=%s | unit=%s", city, unit)
    weather_data_celsius = {
        "istanbul": {"temp": 17.0, "condition": "Partly cloudy"},
        "ankara": {"temp": 14.0, "condition": "Sunny"},
        "izmir": {"temp": 19.0, "condition": "Clear"},
        "london": {"temp": 11.0, "condition": "Rain"},
        "new york": {"temp": 9.0, "condition": "Windy"},
    }

    normalized_city = city.split(",")[0].strip()
    key = normalized_city.lower()
    data = weather_data_celsius.get(key)
    if not data:
        result = f"No weather data found for '{city}'."
        Logger.info("weather_lookup_tool result | city=%s | output=%s", city, result)
        return result

    unit_key = unit.strip().lower()
    if unit_key == "f":
        temp = (data["temp"] * 9 / 5) + 32
        unit_label = "F"
    else:
        temp = data["temp"]
        unit_label = "C"

    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    result = (
        f"{normalized_city.title()} weather as of {now_utc}: "
        f"{temp:.1f}°{unit_label}, {data['condition']}."
    )
    Logger.info(
        "weather_lookup_tool result | city=%s | normalized_city=%s | output=%s",
        city,
        normalized_city,
        result,
    )
    return result


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numeric values.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Sum of `a` and `b`.
    """
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second value from the first.

    Args:
        a: Minuend.
        b: Subtrahend.

    Returns:
        Result of `a - b`.
    """
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numeric values.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        Product of `a` and `b`.
    """
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide one number by another.

    Args:
        a: Dividend.
        b: Divisor. Must not be zero.

    Returns:
        Quotient of `a / b`.

    Raises:
        ValueError: If `b` is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@tool
def square_root(value: float) -> float:
    """Compute square root for a non-negative number.

    Args:
        value: Input value. Must be greater than or equal to zero.

    Returns:
        Square root of `value`.

    Raises:
        ValueError: If `value` is negative.
    """
    if value < 0:
        raise ValueError("Square root is only defined for non-negative values.")
    return math.sqrt(value)
