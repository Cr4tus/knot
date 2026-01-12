import sys

from datetime import datetime, timedelta

from simulator.utils import get_project_root, load_config, configure_logger
from simulator.api import fetch_portfolio_data
from simulator.engine import simulator_factory
from simulator.processor import PortfolioProcessor, StressTester
from simulator.visualizer import Visualizer

def main():
    # 1. Initial Logger Setup
    logger = configure_logger(level="INFO")
    logger.info("Initializing Quantitative Risk Simulator...")

    try:
        project_root_path = get_project_root()

        # 2. Config Setup
        logger.info("Loading configurations...")
        cfg = load_config(project_root_path / "config.yaml")
        configure_logger(level=cfg.logging.level, fmt=cfg.logging.format)

        # 3. Unified Data Acquisition
        # Fetch full history once to avoid multiple API calls.
        logger.info("Fetching historical data...")
        data = fetch_portfolio_data(
            tickers=cfg.portfolio.stocks,
            benchmarks=cfg.portfolio.benchmarks,
            start_date="2019-01-01",
            end_date=cfg.dates.end
        )
        logger.info(f"Successfully retrieved {len(data)} days of data for {len(data.columns)} symbols.")

        # 4. Data Processing for Simulation & Visualization
        simulation_data = data.loc[cfg.dates.start : cfg.dates.end]
        simulation_assets_returns = simulation_data[cfg.portfolio.stocks].pct_change().dropna()
        simulation_portfolio_return = (simulation_assets_returns * cfg.portfolio.weights).sum(axis=1)
        simulation_benchmark_returns = simulation_data[cfg.portfolio.benchmarks].pct_change().dropna()

        target_date = (
            datetime.strptime(cfg.dates.end, cfg.dates.format)
            + timedelta(days=cfg.simulation.days_ahead)
        ).strftime(cfg.dates.format)

        # 5. Simulation
        logger.info(f"Running {cfg.simulation.type} simulation for {cfg.simulation.days_ahead} days...")
        simulation_engine = simulator_factory(cfg.simulation.type, simulation_data, cfg)
        raw_paths = simulation_engine.run(
            n_simulations=cfg.simulation.n_simulations,
            days=cfg.simulation.days_ahead
        )

        # 6. Quantitative Analysis
        processor = PortfolioProcessor(
            simulation_results=raw_paths,
            ticker_names=cfg.portfolio.stocks,
            weights=cfg.portfolio.weights,
        )

        portfolio_paths = processor.get_portfolio_paths()
        metrics = processor.calculate_risk_metrics(portfolio_paths)

        # 7. Stress Testing
        tester = StressTester(data, cfg.portfolio.weights, cfg.portfolio.stocks)
        portfolio_stress, bench_stress = tester.run_stress_tests(cfg.portfolio.benchmarks)

        # 8. Visualization Suite
        logger.info("Generating visual resources...")
        viz = Visualizer(export_dir=project_root_path / cfg.output.export_dir)
        
        # Core Risk Visuals
        viz.plot_correlation_heatmap(simulation_data.pct_change().dropna())
        viz.plot_simulation_paths(portfolio_paths)
        viz.plot_return_distribution(portfolio_paths[:, -1] - 1, metrics['var_95'], target_date)
        viz.plot_stress_test(portfolio_stress, bench_stress)
        viz.plot_benchmark_comparison(simulation_portfolio_return, simulation_benchmark_returns)

        logger.info(f"Simulation workflow complete. Results exported to: {cfg.output.export_dir}")

    except Exception as e:
        logger.critical(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
