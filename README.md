# Portfolio Risk Simulator

A quantitative tool for simulating portfolio trajectories, performing historical stress tests, and generating automated PDF risk reports. This project utilizes stochastic calculus and Monte Carlo methods to model market uncertainty.

## 🚀 Getting Started
### Prerequisites
- Python 3.10 or higher
- `pip` or `poetry` (this guide uses standard venv + pip)

### Installation

#### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/portfolio-risk-simulator.git
cd portfolio-risk-simulator
```

#### 2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies:

```bash
pip install -e .
```

#### 4. Run the application:

```bash
# using the defined shortcut within pyproject.toml
simulate

# or as a normal python script
python src/simulator/main.py
```

## 📂 Project Structure

```bash
├── config.yaml          # Simulation parameters and scenario definitions
├── pyproject.toml       # Project metadata and dependencies
├── src/
│   └── simulator/
│       ├── main.py      # Entry point (Workflow Orchestration)
│       ├── data/        # Data fetching (API) and Pydantic Models
│       ├── engine/      # Stochastic Engines (GBM, Jump Diffusion, etc.)
│       └── utils/       # Math Processing, Visualization, and PDF Reporting
└── output/              # Generated visuals and final PDF reports
```

## ⚙️ Configuration (config.yaml)
The system is entirely data-driven. Key sections include:
- **Portfolio**: Define tickers, target weights, and benchmark indices.
- **Simulation**: Set the number of iterations (n_simulations) and the calibration_window used to estimate historical drift and volatility.
- **Stress Tests**: Define historical date ranges to evaluate how the current portfolio would have performed during past market crashes.
- **Output**: Toggle clear_on_start to maintain a clean workspace and configure asset DPI for high-resolution report charts.

## 🔬 Mathematical Framework
    
### 1. Geometric Brownian Motion (GBM)
The foundational model for asset prices, assuming a constant drift and volatility. It follows the SDE:
$$dS_t=μS_tdt+σS_tdW_t$$
Where $W_t$ is a standard Wiener process.

### 2. Merton Jump Diffusion
Extends GBM by adding a Poisson process to account for *"black swan"* events or sudden market shocks.
$$dS_t=(μ−λκ)S_tdt+σS_tdW_t+(Y−1)S_tdN_t$$
- λ: Frequency of jumps.
- $dN_t$: Poisson process.
- $Y−1$: Random jump size (log-normally distributed).

### 3. Risk Metrics
The system validates and calculates institutional risk figures using Pydantic models:
- VaR (Value at Risk): The 5th percentile of terminal returns (95% confidence).
- CVaR (Conditional VaR): The average of all returns in the worst 5% tail.
- Max Drawdown: The maximum observed peak-to-trough decline across simulated paths.

## 📊 Outputs & Artifacts
Upon execution, the system generates a structured output/ directory:
1. **Stochastic Trajectories**: Visualizes potential future portfolio paths.
2. **Probability Density**: A histogram of terminal returns with VaR/CVaR overlays.
3. **Stress Scenarios**: A comparative bar chart showing Portfolio vs. Benchmark performance during crisis periods.
4. **The Report**: A unified portfolio_risk_report.pdf containing all metadata, mathematical evidence, and visual plots.
