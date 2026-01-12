import datetime

from fpdf import FPDF

from simulator.data.simulation_engine_result import SimulationEngineResult


class RiskReporter(FPDF):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.set_auto_page_break(auto=True, margin=15)


    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f"Portfolio Risk Report - {self.cfg.portfolio.stocks}", 0, 0, 'R')
            self.ln(10)


    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


    def add_title_page(self):
        self.add_page()
        self.set_font('helvetica', 'B', 24)
        self.set_y(60)
        self.cell(0, 20, "Quantitative Risk Analysis Report", 0, 1, 'C')
        self.set_font('helvetica', '', 16)
        self.cell(0, 10, f"Portfolio: {', '.join(self.cfg.portfolio.stocks)}", 0, 1, 'C')
        self.ln(20)
        self.set_font('helvetica', 'I', 12)
        self.cell(0, 10, f"Generated on: {datetime.date.today().strftime('%B %d, %Y')}", 0, 1, 'C')


    def add_introduction(self, extended_start_date: str, start_date: str, end_date: str):
        self.add_page()
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, "1. Executive Summary & Methodology", align='C')
        self.ln(20)
        self.set_font('helvetica', '', 11)
        intro_text = (
            "This report provides a comprehensive stress test and probabilistic simulation of the target portfolio. "
            "We employ a 'Multi-Engine' approach, comparing three distinct stochastic models to capture different "
            "market regimes: standard Monte Carlo, Geometric Brownian Motion (GBM), and Merton Jump Diffusion. "
            f"\n\nHistorical data calibration period: {extended_start_date} to {end_date}."
            f"\nSimulation period: {start_date} to {end_date}."
        )
        self.multi_cell(0, 7, intro_text)

        self.ln(5)
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, "Primary Risk Metric: Value at Risk (VaR)", 0, 1, 'L')
        self.set_font('times', 'I', 12)
        self.cell(0, 7, "VaR_alpha = inf { l in R : P(L > l) <= 1 - alpha }")


    def add_correlation_heatmap_visual_and_simulations_comparison(self,
                                   correlation_heatmap_visual_filepath: str,
                                   engine_to_results_map: dict[str, SimulationEngineResult]):
        self.add_page()
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, "2. Multi-Engine Simulation Analysis", 0, 1, 'L')
        
        # Correlation Heatmap
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, "Asset Correlations", 0, 1, 'L')
        self.image(correlation_heatmap_visual_filepath, x=10, w=100)
        self.ln(5)
        
        # Comparison Table
        self.set_y(self.get_y() + 10)
        self.set_font('helvetica', 'B', 11)
        self.cell(40, 10, "Engine", 1)
        self.cell(50, 10, "Exp. Return (%)", 1)
        self.cell(50, 10, "95% VaR (%)", 1)
        self.cell(50, 10, "Max Drawdown (%)", 1)
        self.ln()
        
        self.set_font('helvetica', '', 11)
        for engine, data in engine_to_results_map.items():
            self.cell(40, 10, engine.replace('_', ' ').title(), 1)
            self.cell(50, 10, f"{data.metrics.expected_return:.2%}", 1)
            self.cell(50, 10, f"{data.metrics.var_95:.2%}", 1)
            self.cell(50, 10, f"{data.metrics.max_drawdown:.2%}", 1)
            self.ln()


    def add_pages_for_engines_results(self, engine_to_results_map: dict[str, SimulationEngineResult]):
        for engine, data in engine_to_results_map.items():
            self.add_page()
            self.set_font('helvetica', 'B', 14)
            self.cell(0, 10, f"Model Detail: {engine.replace('_', ' ').title()}", 0, 1, 'L')
            self.image(str(data.simulation_visual_filepath), x=10, w=180)
            self.ln(5)
            self.image(str(data.return_distribution_visual_filepath), x=10, w=180)


    def add_portfolio_vs_benchmarks_and_stress_test_visuals(self,
                                                            portfolio_vs_benchmarks_visual_filepath: str,
                                                            stress_test_visual_filepath: str):
        self.add_page()
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, "3. Historical Stress Testing & Benchmarking", 0, 1, 'L')
        
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, "Performance vs. Benchmarks", 0, 1, 'L')
        self.image(portfolio_vs_benchmarks_visual_filepath, x=10, w=180)
        
        self.ln(10)
        self.cell(0, 10, "Crisis Scenario: 2020 COVID Crash", 0, 1, 'L')
        self.image(stress_test_visual_filepath, x=10, w=180)
