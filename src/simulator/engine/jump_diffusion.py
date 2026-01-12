import numpy as np
import logging
from simulator.engine.base import BaseSimulator


logger = logging.getLogger(__name__)


class JumpDiffusionSimulator(BaseSimulator):
    def run(self, n_simulations: int, days: int):
        """
        Merton Jump-Diffusion: GBM + Poisson Jumps for tail risk.
        """

        logger.info("Starting Merton Jump-Diffusion simulation.")

        # Parameters (usually calibrated or set in config)
        # For now, we use historical vol + config-based jump settings
        asset_data = self.data[self.tickers]
        returns = np.log(asset_data / asset_data.shift(1)).dropna()
        
        mu = returns.mean().values
        sigma = returns.std().values
        
        # Jump settings (Lambda = jumps per year, J_mu = jump size, J_sigma = jump vol)
        lam = self.config.simulation.get('jump_lambda', 0.1) 
        j_mu = self.config.simulation.get('jump_mu', -0.05)
        j_sigma = self.config.simulation.get('jump_sigma', 0.1)

        results = []


        for _ in range(n_simulations):
            # 1. Standard GBM Component
            random_shocks = np.random.normal(0, 1, (days, len(self.tickers)))
            gbm_part = (mu - 0.5 * sigma**2) + (sigma * random_shocks)
            
            # 2. Jump Component (Poisson process)
            # If a jump occurs, it adds a normally distributed shock
            jumps_occurred = np.random.poisson(lam / 252, (days, len(self.tickers)))
            jump_part = jumps_occurred * np.random.normal(j_mu, j_sigma, (days, len(self.tickers)))
            
            total_log_return = gbm_part + jump_part
            price_paths = np.cumprod(np.exp(total_log_return), axis=0)
            results.append(price_paths)


        return np.array(results)