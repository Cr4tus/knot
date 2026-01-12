import sys

from datetime import datetime, timedelta

from simulator.utils.functions import get_project_root, load_config, configure_logger
from simulator.data.api import fetch_portfolio_data
from simulator.engine import simulator_factory
from simulator.utils.processor import PortfolioProcessor, StressTester
from simulator.utils.visualizer import Visualizer
from simulator.utils.reporter import RiskReporter
from simulator.data.model.simulation_engine_result import SimulationEngineResult


def main():
    # 1. Default Logger Setup
    logger = configure_logger(level="INFO")

    try:
        project_root_path = get_project_root()

        # 2. Config Setup
        logger.info("Loading configurations...")
        config = load_config(project_root_path / "config.yaml")
        configure_logger(
            level=config.project.logging.level,
            fmt=config.project.logging.format
        )

        output_directory = project_root_path / config.output.directory

        # 3. Unified Data Acquisition
        # Fetch full history once to avoid multiple API calls.
        logger.info(
            f"Initiating data download for Stocks: {config.portfolio.stocks} | "
            f"Benchmarks: {config.portfolio.benchmarks}"
        )

        # 4. Data Processing for Simulation & Visualization
        simulation_data = fetch_portfolio_data(
            tickers=set(config.portfolio.stocks),
            benchmarks=set(config.portfolio.benchmarks),
            start_date=config.simulation.calibration_window.start,
            end_date=config.simulation.calibration_window.end,
            dates_format=config.project.dates_format
        )
        simulation_assets_returns = simulation_data[config.portfolio.stocks].pct_change().dropna()
        simulation_portfolio_return = (simulation_assets_returns * config.portfolio.weights).sum(axis=1)
        simulation_benchmark_returns = simulation_data[config.portfolio.benchmarks].pct_change().dropna()
        target_date = (
            datetime.strptime(config.simulation.calibration_window.end, config.project.dates_format)
            + timedelta(days=config.simulation.days_ahead)
        ).strftime(config.project.dates_format)

        # 5. Simulations
        engine_to_results_map: dict[str, SimulationEngineResult] = dict()
        for engine in config.simulation.active_engines:
            logger.info(f"Running {engine} simulation...")

            simulation_engine = simulator_factory(engine, simulation_data, config)
            raw_paths = simulation_engine.run(
                config.simulation.n_simulations, config.simulation.days_ahead)

            processor = PortfolioProcessor(
                simulation_results=raw_paths,
                ticker_names=config.portfolio.stocks,
                weights=config.portfolio.weights,
            )
            paths = processor.get_portfolio_paths()
            metrics = processor.calculate_risk_metrics(paths)

            # Generate engine-specific visuals
            logger.info(f"Generating visual resources for {engine} simulation...")
            visualizer = Visualizer(
                directory=output_directory / engine,
                config=config.output.visualizer,
            )

            simulation_visual_filepath = \
                visualizer.plot_simulation_paths(portfolio_paths=paths)

            return_distribution_visual_filepath = \
                visualizer.plot_return_distribution(
                    final_returns=paths[:, -1] - 1,
                    var_95=metrics.var_95,
                    target_date=target_date,
                )

            # Saving results for PDF generation
            engine_to_results_map[engine] = SimulationEngineResult(
                metrics=metrics,
                simulation_visual_filepath=simulation_visual_filepath,
                return_distribution_visual_filepath=return_distribution_visual_filepath,
            )

        # 6. Stress Testing
        logger.info("Running stress scenarios...")
        tester = StressTester(
            config=config.stress_tests,
            dates_format=config.project.dates_format
        )
        portfolio_stress, bench_stress = tester.run_stress_tests(
            weights=config.portfolio.weights,
            tickers=set(config.portfolio.stocks),
            benchmark_tickers=set(config.portfolio.benchmarks),
        )

        # 7. Visualization
        logger.info("Generating other visual resources...")
        visualizer = Visualizer(
            directory=output_directory,
            config=config.output.visualizer
        )

        correlation_heatmap_visual_filepath = \
            visualizer.plot_correlation_heatmap(
                data=simulation_data.pct_change().dropna(),
            )
        
        stress_test_visual_filepath = \
            visualizer.plot_stress_tests(
                portfolio_results=portfolio_stress,
                benchmark_results=bench_stress,
            )
        
        portfolio_vs_benchmark_visual_filepath = \
            visualizer.plot_benchmark_comparison(
                simulation_portfolio_return,
                simulation_benchmark_returns,
            )

        # 8. PDF Generation
        if config.output.report.generate:
            logger.info("Generating PDF report...")

            reporter = RiskReporter(config)
            reporter.add_title_page()
            reporter.add_introduction(
                start_date=config.simulation.calibration_window.start,
                end_date=config.simulation.calibration_window.end,
            )
            reporter.add_correlation_heatmap_visual_and_simulations_comparison(
                correlation_heatmap_visual_filepath=str(correlation_heatmap_visual_filepath),
                engine_to_results_map=engine_to_results_map,
            )
            reporter.add_pages_for_engines_results(
                engine_to_results_map=engine_to_results_map,
            )
            reporter.add_portfolio_vs_benchmarks_and_stress_tests_visuals(
                portfolio_vs_benchmarks_visual_filepath=str(portfolio_vs_benchmark_visual_filepath),
                stress_test_visual_filepath=str(stress_test_visual_filepath),
            )

            reporter.output(str(output_directory / config.output.report.filename))

    except Exception as e:
        logger.critical(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
