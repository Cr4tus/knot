import numpy as np
import logging
from simulator.engine.base import BaseSimulator


logger = logging.getLogger(__name__)


class GeometricBrownianSimulator(BaseSimulator):
    def run(self, n_simulations: int, days: int):
        """
        Runs simulation by sampling historical return days.
        """

        logger.info(f"Starting Historical Geometric Brownian Motion on {len(self.tickers)} assets.")

        returns = self.data[self.tickers].pct_change().dropna().values
        results = []


        for _ in range(n_simulations):
            indices = np.random.choice(len(returns), size=days, replace=True)
            sampled_returns = returns[indices]
            
            # price_path = (1 + r1) * (1 + r2) ...
            price_paths = np.cumprod(1 + sampled_returns, axis=0)
            results.append(price_paths)


        return np.array(results)