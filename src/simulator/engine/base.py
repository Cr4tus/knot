import logging
import pandas as pd

from abc import ABC, abstractmethod
from typing import Any
from box import Box


logger = logging.getLogger(__name__)


class BaseSimulator(ABC):
    def __init__(self, data: pd.DataFrame, config: Box):
        """
        Initializes the simulator with historical data and config.
        """

        self.data = data
        self.config = config
        self.tickers = config.portfolio.stocks
        self.benchmarks = config.portfolio.benchmarks


    @abstractmethod
    def run(self, n_simulations: int, days: int) -> Any:
        """
        Execute the simulation logic.
        """

        pass