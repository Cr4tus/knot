import logging

from simulator.engine.monte_carlo import MonteCarloSimulator
from simulator.engine.bootstrap import BootstrapSimulator
from simulator.engine.jump_diffusion import JumpDiffusionSimulator


logger = logging.getLogger(__name__)


def simulator_factory(simulation_type: str, data, config):
    """
    Returns an instance of the requested simulator type.
    """

    strategies = {
        "monte_carlo": MonteCarloSimulator,
        "bootstrap": BootstrapSimulator,
        "jump_diffusion": JumpDiffusionSimulator
    }

    if simulation_type not in strategies:
        logger.error(f"Unsupported simulation type: {simulation_type}")
        raise ValueError(f"Strategy {simulation_type} not found.")

    logger.info(f"Initializing {simulation_type} engine.")
    return strategies[simulation_type](data, config)