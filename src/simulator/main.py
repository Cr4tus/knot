import sys

from datetime import datetime, timedelta

from simulator.utils import get_project_root, load_config, configure_logger
from simulator.api import fetch_portfolio_data
from simulator.engine import simulator_factory
from simulator.processor import PortfolioProcessor, StressTester
from simulator.visualizer import Visualizer
from simulator.reporter import RiskReporter
from simulator.data.simulation_engine_result import SimulationEngineResult


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
            start_date=cfg.dates.extended,
            end_date=cfg.dates.end,
            date_format=cfg.dates.format
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

        # 5. Simulations
        engine_to_results_map: dict[str, SimulationEngineResult] = dict()
        for engine_type in cfg.simulation.active:
            # Actual simulation run
            simulation_engine = simulator_factory(engine_type, simulation_data, cfg)
            raw_paths = simulation_engine.run(cfg.simulation.n_simulations, cfg.simulation.days_ahead)

            processor = PortfolioProcessor(
                simulation_results=raw_paths,
                ticker_names=cfg.portfolio.stocks,
                weights=cfg.portfolio.weights,
            )
            paths = processor.get_portfolio_paths()
            metrics = processor.calculate_risk_metrics(paths)

            # Generate engine-specific visuals
            logger.info(f"Generating visual resources for {engine_type} simulation...")
            visualizer = Visualizer(
                directory=project_root_path / cfg.output.directory / engine_type,
                config=cfg.output.visualizer
            )

            simulation_visual_filepath = \
                visualizer.plot_simulation_paths(portfolio_paths=paths)

            return_distribution_visual_filepath = \
                visualizer.plot_return_distribution(
                    final_returns=paths[:, -1] - 1,
                    var_95=metrics.var_95,
                    target_date=target_date
                )

            # Saving results for PDF generation
            engine_to_results_map[engine_type] = SimulationEngineResult(
                metrics=metrics,
                simulation_visual_filepath=simulation_visual_filepath,
                return_distribution_visual_filepath=return_distribution_visual_filepath
            )

        # 7. Stress Testing
        tester = StressTester(data, cfg.portfolio.weights, cfg.portfolio.stocks)
        portfolio_stress, bench_stress = tester.run_stress_tests(cfg.portfolio.benchmarks)

        # 8. Visualization
        logger.info("Generating other visual resources...")
        visualizer = Visualizer(
            directory=project_root_path / cfg.output.directory,
            config=cfg.output.visualizer
        )
        
        # Core Risk Visuals
        correlation_heatmap_visual_filepath = \
            visualizer.plot_correlation_heatmap(
                data=simulation_data.pct_change().dropna()
            )
        
        stress_test_visual_filepath = \
            visualizer.plot_stress_test(
                portfolio_results=portfolio_stress,
                benchmark_results=bench_stress
            )
        
        portfolio_vs_benchmark_visual_filepath = \
            visualizer.plot_benchmark_comparison(
                simulation_portfolio_return,
                simulation_benchmark_returns
            )

        logger.info(f"Simulation workflow complete. Results exported to: {cfg.output.directory}")

        # 9. PDF Generation
        if cfg.output.report.generate:
            logger.info("Generating PDF report...")

            reporter = RiskReporter(cfg)
            reporter.add_title_page()
            reporter.add_introduction(
                extended_start_date=cfg.dates.extended,
                start_date=cfg.dates.start,
                end_date=cfg.dates.end,
            )
            reporter.add_correlation_heatmap_visual_and_simulations_comparison(
                correlation_heatmap_visual_filepath=str(correlation_heatmap_visual_filepath),
                engine_to_results_map=engine_to_results_map
            )
            reporter.add_pages_for_engines_results(
                engine_to_results_map=engine_to_results_map
            )
            reporter.add_portfolio_vs_benchmarks_and_stress_test_visuals(
                portfolio_vs_benchmarks_visual_filepath=str(portfolio_vs_benchmark_visual_filepath),
                stress_test_visual_filepath=str(stress_test_visual_filepath),
            )

            reporter.output(str(project_root_path / cfg.output.report.filename))

    except Exception as e:
        logger.critical(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
