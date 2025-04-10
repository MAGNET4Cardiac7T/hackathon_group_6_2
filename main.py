from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer
from src.data import Simulation, CoilConfig

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 100) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    direction = "maximize"
    optimizer = DummyOptimizer(cost_function=cost_function)
    best_score = -np.inf if direction == "maximize" else np.inf
        
    for i in range(3):
        config = optimizer.optimize(simulation)
        
        simulation_data = simulation(config)

        score = cost_function(simulation_data)
        
        if score > best_score:
            best_score = score
            best_coil_config = config
    return best_coil_config
