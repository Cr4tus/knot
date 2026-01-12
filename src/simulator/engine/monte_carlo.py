import numpy as np
import logging
from simulator.engine.base import BaseSimulator


logger = logging.getLogger(__name__)


class MonteCarloSimulator(BaseSimulator):
    def run(self, n_simulations: int, days: int):
        """
        Runs Monte Carlo simulation using Cholesky decomposition on asset returns.
        """

        logger.info(f"Starting Monte Carlo on {len(self.tickers)} assets.")

        # Use log returns for GBM math
        asset_data = self.data[self.tickers]
        returns = np.log(asset_data / asset_data.shift(1)).dropna()
        
        avg_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov_matrix)

        results = []


        for _ in range(n_simulations):
            # Generate correlated random walks
            drift = avg_returns
            random_shocks = np.random.normal(0, 1, (days, len(self.tickers)))
            correlated_shocks = random_shocks @ L.T
            
            daily_growth = np.exp(drift + correlated_shocks)
            price_paths = np.cumprod(daily_growth, axis=0)
            results.append(price_paths)


        return np.array(results)