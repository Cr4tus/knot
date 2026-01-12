import sys

from simulator.engine.base import BaseSimulator
# Those imports are needed in order to see the classes
from simulator.engine.geometric_brownian import GeometricBrownianSimulator
from simulator.engine.jump_diffusion import JumpDiffusionSimulator
from simulator.engine.monte_carlo import MonteCarloSimulator


def simulator_factory(simulation_type: str, data, config) -> BaseSimulator:
    current_module = sys.modules[__name__]
    print(current_module)
    class_name = simulation_type.title().replace("_", "") + "Simulator"
    simulator_class = getattr(current_module, class_name, None)

    if simulator_class is None:
        raise ValueError(f"Could not find {class_name} within the engine module.")

    return simulator_class(data, config)