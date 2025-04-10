from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer
from src.data import Simulation, CoilConfig
import time
import numpy as np
import torch
import concurrent.futures
from functools import partial

def trial(_, simulation, cost_function):
    # Create one trial by optimizing and evaluating the simulation
    optimizer = DummyOptimizer(cost_function=cost_function)
    config = optimizer.optimize(simulation)
    simulation_data = simulation(config)
    score = cost_function(simulation_data)
    return score, config

def run_trials_parallel(simulation: "Simulation", 
                        cost_function: "BaseCost",
                        timeout: int = 100) -> "CoilConfig":
    best_score = float('-inf')
    best_coil_config = None

    # Create a partially applied version of trial with simulation & cost_function included
    trial_partial = partial(trial, simulation=simulation, cost_function=cost_function)
    
    # Submit 25 parallel tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(trial_partial, range(4))
    
    # Retrieve the best result
    for score, config in results:
        if score > best_score:
            best_score = score
            best_coil_config = config

    return best_coil_config

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 100, iterations: int = 25) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    direction = "maximize"
    optimizer = DummyOptimizer(cost_function=cost_function, lr=0.2)
    best_score = -torch.inf if direction == "maximize" else torch.inf
    
    #scores = []
    iteration_times = []
    max_duration = 5*60-20
    start_time = time.time()

    for i in range(iterations):
        current_time = time.time()
        elapsed_time = current_time - start_time

        max_iter_time = np.max(iteration_times) if iteration_times else 0
        
        if elapsed_time > max_duration-max_iter_time:
            print(f"Max duration reached: {elapsed_time:.2f} seconds, iteration: ", i)
            break
        
        iter_start = time.time()
        #optimizer = DummyOptimizer(cost_function=cost_function, lr=(i+1)*0.01)
        config = optimizer.optimize(simulation)
        
        simulation_data = simulation(config)

        score = cost_function(simulation_data)
        
        if score > best_score:
            best_score = score
            best_coil_config = config
        
        iter_end = time.time()
        iter_duration = iter_end - iter_start
        iteration_times.append(iter_duration)
        #scores.append(score.detach().numpy())

    #plt.plot(scores)
    #plt.show()
    return best_coil_config
