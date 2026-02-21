from datetime import date, timedelta
from typing import Dict, Tuple, List


POPULAR_STOCKS: Dict[str, str] = {
    "Reliance Industries (NSE)": "RELIANCE.NS",
    "HDFC Bank (NSE)": "HDFCBANK.NS",
    "TCS (NSE)": "TCS.NS",
    "Infosys (NSE)": "INFY.NS",
    "ICICI Bank (NSE)": "ICICIBANK.NS",
    "State Bank of India (NSE)": "SBIN.NS",
    "Axis Bank (NSE)": "AXISBANK.NS",
    "Bharti Airtel (NSE)": "BHARTIARTL.NS",
    "ITC (NSE)": "ITC.NS",
    "Kotak Mahindra Bank (NSE)": "KOTAKBANK.NS",
}


def get_date_range(preset: str) -> Tuple[date, date]:
    """Return (start_date, end_date) based on a simple preset string."""
    today = date.today()

    presets = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
    }
    days = presets.get(preset, 180)
    return today - timedelta(days=days), today


def chunk_list(items: List[str], max_len: int) -> List[List[str]]:
    """Chunk a list into sub-lists with at most max_len items."""
    return [items[i : i + max_len] for i in range(0, len(items), max_len)]

