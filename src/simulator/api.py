import logging
import pandas as pd
import yfinance as yf

from simulator.utils import validate_date_interval


logger = logging.getLogger(__name__)


def fetch_portfolio_data(
        tickers: list[str],
        benchmarks: list[str],
        start_date: str,
        end_date: str
) -> pd.DataFrame:
    """
    Fetches closing prices for a list of tickers and multiple benchmarks.
    """

    validate_date_interval(start_date, end_date, date_format="%Y-%m-%d")

    all_symbols = tickers + benchmarks
    logger.info(f"Initiating data download for Stocks: {tickers} | Benchmarks: {benchmarks}")

    try:
        data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)

        if data is None or data.empty:
            raise ValueError(f"No data found for symbols {all_symbols} in the requested range.")

        close_data = data['Close']
        clean_data = close_data.dropna()

        if clean_data.empty:
            raise ValueError("No complete data found after dropping missing values.")

        if isinstance(clean_data, pd.Series):
            clean_data = clean_data.to_frame()

        return clean_data

    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise