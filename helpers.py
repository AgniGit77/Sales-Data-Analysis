"""
Helper / utility functions for formatting and display.
"""

def format_currency(value: float) -> str:
    """Format number as currency string (e.g., $1.2M, $45.3K)."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:,.0f}"


def format_number(value: float) -> str:
    """Format large numbers with K/M suffix."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def format_percentage(value: float) -> str:
    """Format as percentage with sign."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def trend_icon(value: float) -> str:
    """Return trend arrow based on value."""
    if value > 0:
        return "↑"
    elif value < 0:
        return "↓"
    return "→"


def trend_color(value: float) -> str:
    """Return CSS color class based on trend."""
    if value > 0:
        return "#00e676"  # green
    elif value < 0:
        return "#ff5252"  # red
    return "#ffd740"      # amber


def get_quarter(month: int) -> str:
    """Return quarter string from month number."""
    return f"Q{(month - 1) // 3 + 1}"


def get_greeting() -> str:
    """Return time-based greeting."""
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"
