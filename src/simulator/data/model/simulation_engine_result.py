from pydantic import BaseModel, FilePath

from simulator.data.model.risk_metrics import RiskMetrics


class SimulationEngineResult(BaseModel):
    """
    Container for a specific simulation engine's output.
    Links mathematical metrics with their corresponding visual assets' filepaths.
    """
    metrics: RiskMetrics
    simulation_visual_filepath: FilePath
    return_distribution_visual_filepath: FilePath
