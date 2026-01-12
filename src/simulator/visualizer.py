import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from box import Box


class Visualizer:
    def __init__(self, directory: Path, config: Box):
        self.cfg = config
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="darkgrid")
    

    def plot_correlation_heatmap(self, data: pd.DataFrame):
        """
        Generates a heatmap of asset correlations.
        Essential for identifying 'Hidden Risk' where assets move in lockstep.
        """

        plt.figure(figsize=(10, 8))
        
        # Calculate Pearson correlation matrix
        corr_matrix = data.corr()
        
        # Create a mask to hide the upper triangle (it's a mirror image anyway)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw the heatmap
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap=sns.color_palette("magma", as_cmap=True), 
            center=0,
            square=True,
            linewidths=.5, 
            cbar_kws={"shrink": .8}
        )
        
        plt.title("Asset Correlation Matrix (Historical Data)")
        
        path = self.directory / self.cfg.filenames.correlation_heatmap
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path


    def plot_simulation_paths(self, portfolio_paths: np.ndarray):
        """Generates a fan chart of all simulated paths."""

        plt.figure(figsize=(12, 6))
        days = np.arange(portfolio_paths.shape[1])
        
        # Plot a subset of paths for clarity
        for i in range(min(100, portfolio_paths.shape[0])):
            plt.plot(days, portfolio_paths[i, :], color='royalblue', alpha=0.05)
            
        # Highlight percentiles
        plt.plot(days, np.median(portfolio_paths, axis=0), color='royalblue', label='Median', linewidth=2)
        plt.plot(days, np.percentile(portfolio_paths, 95, axis=0), color='lime', linestyle='--', label='95th Percentile')
        plt.plot(days, np.percentile(portfolio_paths, 5, axis=0), color='red', linestyle='--', label='5th Percentile')
        
        plt.title("Simulated Portfolio Cumulative Returns")
        plt.xlabel("Days Ahead")
        plt.ylabel("Value (Starting at 1.0)")
        plt.legend()
        
        path = self.directory / self.cfg.filenames.simulation_paths
        plt.savefig(path, dpi=300)
        plt.close()

        return path


    def plot_return_distribution(self, final_returns: np.ndarray, var_95: float, target_date: str):
        plt.figure(figsize=(10, 6))
        # Convert to % for readability
        data_pct = final_returns * 100
        
        sns.histplot(data_pct, kde=True, color='skyblue', bins=50, stat="probability")
        plt.axvline(var_95 * 100, color='#ff4d4d', linestyle='--', linewidth=2, 
                    label=f'95% VaR: {var_95:.1%}')
        
        plt.title(f"End-of-Period Return Probability\nTarget Date: {target_date}")
        plt.xlabel("Total Cumulative Return (%)")
        plt.ylabel("Probability")
        plt.legend()
        
        path = self.directory / self.cfg.filenames.return_distribution
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path
    

    def plot_benchmark_comparison(self, historical_returns: pd.Series, benchmarks_returns: pd.DataFrame):
        """
        Compares the actual portfolio performance against all benchmarks.
        This is the 'Reality Check' chart.
        """

        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative growth starting from 1.0
        portfolio_growth = (1 + historical_returns).cumprod()
        
        # 1. Plot the actual portfolio
        plt.plot(portfolio_growth.index, portfolio_growth, label='Actual Portfolio', color='gold', linewidth=2.5)
        
        # 2. Plot all benchmarks in the DataFrame
        # We use a color palette to ensure different benchmarks are distinguishable
        colors = sns.color_palette("husl", len(benchmarks_returns.columns))
        
        for i, column in enumerate(benchmarks_returns.columns):
            benchmark_growth = (1 + benchmarks_returns[column]).cumprod()
            plt.plot(
                benchmark_growth.index, 
                benchmark_growth, 
                label=f'Benchmark ({column})', 
                linestyle='--', 
                alpha=0.7,
                color=colors[i]
            )
        
        plt.title("Historical Performance vs. Benchmarks")
        plt.ylabel("Cumulative Growth (1.0 = Start)")
        plt.legend()
        
        path = self.directory / self.cfg.filenames.portfolio_vs_benchmark
        plt.savefig(path, dpi=300)
        plt.close()

        return path
    

    def plot_stress_test(self, portfolio_results: dict, benchmark_results: dict):
        if not portfolio_results: return
        
        plt.figure(figsize=(12, 8))
        
        scenarios = list(portfolio_results.keys())
        scenario = scenarios[0] # Focus on the COVID scenario
        
        # Prepare labels and values
        names = ["Portfolio"] + list(benchmark_results[scenario].keys())
        values = [portfolio_results[scenario] * 100] + [v * 100 for v in benchmark_results[scenario].values()]
        
        # Professional color palette
        colors = ['#ff4d4d'] + sns.color_palette("Greys_r", len(names)-1)
        
        bars = plt.barh(names, values, color=colors)
        
        # NEW: Center labels inside the bars
        for bar in bars:
            width = bar.get_width()
            # We place the text at width / 2 to center it horizontally
            plt.text(
                width / 2, 
                bar.get_y() + bar.get_height() / 2, 
                f'{width:.1f}%', 
                va='center', 
                ha='center', # Center alignment
                color='black', 
                fontweight='bold',
                fontsize=10
            )
        
        plt.axvline(0, color='white', linewidth=1)
        plt.title(f"Crisis Impact Comparison: {scenario}")
        plt.xlabel("Total Percentage Return (%)")
        
        # Clean up the look
        plt.gca().invert_yaxis() # Portfolio at the top
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        path = self.directory / self.cfg.filenames.stress_test
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return path
