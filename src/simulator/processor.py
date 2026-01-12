import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


class PortfolioProcessor:
    def __init__(self, simulation_results: np.ndarray, ticker_names: list, weights: list):
        """
        Initializes the processor with simulation paths and portfolio weights.
        """

        self.sim_results = simulation_results  # Shape: (Sims, Days, Assets)
        self.tickers = ticker_names
        self.weights = np.array(weights)
        
        if len(self.tickers) != len(self.weights):
            raise ValueError("Number of tickers must match number of weights.")
            
        # Normalize weights if they don't sum to 1
        if not np.isclose(self.weights.sum(), 1.0):
            logger.warning("Weights do not sum to 1. Normalizing...")
            self.weights = self.weights / self.weights.sum()


    def get_portfolio_paths(self) -> np.ndarray:
        """
        Collapses multi-asset simulations into total portfolio value paths.
        Returns array of shape (Simulations, Days).
        """

        logger.info("Synthesizing portfolio paths from asset simulations.")

        # Dot product across the asset axis (axis 2)
        # Resulting shape: (n_simulations, n_days)
        return np.dot(self.sim_results, self.weights)


    def calculate_risk_metrics(self, portfolio_paths: np.ndarray, confidence_level: float = 0.95):
        """
        Calculates VaR, CVaR, and Maximum Drawdown across all simulations.
        """

        # Final returns at the end of the simulation period
        final_returns = portfolio_paths[:, -1] - 1
        
        # Value at Risk (VaR)
        var = np.percentile(final_returns, (1 - confidence_level) * 100)
        
        # Conditional VaR (Expected Shortfall)
        cvar = final_returns[final_returns <= var].mean() if any(final_returns <= var) else var
        
        # Maximum Drawdown calculation for each path
        # We calculate the peak of each path and the drop from that peak
        peak = np.maximum.accumulate(portfolio_paths, axis=1)
        drawdowns = (portfolio_paths - peak) / peak
        max_drawdown = np.min(drawdowns)

        metrics = {
            "expected_return": np.mean(final_returns),
            "median_return": np.median(final_returns),
            "var_95": var,
            "cvar_95": cvar,
            "max_drawdown": max_drawdown,
            "volatility": np.std(final_returns)
        }

        logger.info(f"Risk metrics calculated: VaR={var:.2%}, MaxDD={max_drawdown:.2%}")
        return metrics


class StressTester:
    def __init__(self, data: pd.DataFrame, weights: list, tickers: list):
        self.data = data
        self.weights = np.array(weights)
        self.tickers = tickers
        self.scenarios = {
            "2020 COVID Crash": ("2020-02-19", "2020-03-23")
        }

    def run_stress_tests(self, benchmark_tickers: list) -> tuple[dict, dict]:
        """
        Runs scenarios for the portfolio and all provided benchmarks.
        """
        logger.info(f"Running Stress Tests against benchmarks: {benchmark_tickers}")
        portfolio_results = {}
        benchmark_results = {} # Structure: {scenario_name: {ticker: value}}

        returns = self.data.pct_change()

        for name, (start, end) in self.scenarios.items():
            try:
                period_returns = returns.loc[start:end]
                if period_returns.empty: continue

                # 1. Portfolio Performance (with dynamic re-weighting)
                valid_tickers = [t for t in self.tickers if t in period_returns.columns]
                orig_w = np.array([self.weights[self.tickers.index(t)] for t in valid_tickers])
                new_w = orig_w / orig_w.sum()
                
                port_perf = (period_returns[valid_tickers] * new_w).sum(axis=1)
                portfolio_results[name] = (1 + port_perf).cumprod().iloc[-1] - 1
                
                # 2. Benchmarks' Performance
                benchmark_results[name] = {}
                for b_ticker in benchmark_tickers:
                    if b_ticker in period_returns.columns:
                        b_perf = period_returns[b_ticker]
                        benchmark_results[name][b_ticker] = (1 + b_perf).cumprod().iloc[-1] - 1

            except Exception as e:
                logger.error(f"Error in {name}: {e}")

        return portfolio_results, benchmark_results
