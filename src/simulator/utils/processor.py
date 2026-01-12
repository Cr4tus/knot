import logging
import numpy as np

from box import Box

from simulator.data.api import fetch_portfolio_data
from simulator.data.model.risk_metrics import RiskMetrics


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


    def calculate_risk_metrics(self,
                               portfolio_paths: np.ndarray,
                               confidence_level: float = 0.95) -> RiskMetrics:
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

        metrics = RiskMetrics(
            expected_return=float(np.mean(final_returns)),
            median_return=float(np.median(final_returns)),
            var_95=float(var),
            cvar_95=float(cvar),
            max_drawdown=max_drawdown,
            volatility=float(np.std(final_returns))
        )

        logger.info(f"Risk metrics calculated: VaR={var:.2%}, MaxDD={max_drawdown:.2%}")
        return metrics


class StressTester:
    def __init__(self, config: Box, dates_format: str):
        self.config = config
        self.date_format = dates_format

    def run_stress_tests(self, weights: list[float], tickers: set[str],
                        benchmark_tickers: set[str]) -> tuple[dict, dict]:
        portfolio_results = {}
        benchmark_results = {}
        for scenario in self.config.scenarios:
            try:
                # 1. Data fetching
                data = fetch_portfolio_data(
                    tickers=tickers,
                    benchmarks=benchmark_tickers,
                    start_date=scenario.start,
                    end_date=scenario.end,
                    dates_format=self.date_format
                )
                returns = data.pct_change().dropna()
                if returns.empty:
                    continue

                # 2. Portfolio Performance
                # (with dynamic re-weighting/re-normalizing)
                weight_map = dict(zip(tickers, weights))
                valid_tickers = [t for t in tickers if t in returns.columns]
                orig_w = np.array([weight_map[t] for t in valid_tickers])
                new_w = orig_w / orig_w.sum()
                
                portfolio_performance = (returns[valid_tickers] * new_w).sum(axis=1)
                portfolio_results[scenario.name] = (1 + portfolio_performance).cumprod().iloc[-1] - 1
                
                # 3. Benchmarks' Performance
                benchmark_results[scenario.name] = {}
                for b_ticker in benchmark_tickers:
                    if b_ticker in returns.columns:
                        b_perf = returns[b_ticker]
                        benchmark_results[scenario.name][b_ticker] = (1 + b_perf).cumprod().iloc[-1] - 1

            except Exception as e:
                logger.error(
                    f"Error occurred while running "
                    f"{scenario.name} stress scenario: {e}"
                )

        return portfolio_results, benchmark_results
