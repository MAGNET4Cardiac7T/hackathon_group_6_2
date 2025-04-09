from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from tqdm import trange


class DummyOptimizer(BaseOptimizer):
    """
    AdamOptimizer uses the Adam optimizer to adjust coil configuration parameters.
    
    It optimizes over an unconstrained vector `u` of size 16 (first 8 for phases, next 8 for amplitudes)
    and maps these to the appropriate ranges via a differentiable transform:
      - phases = 2π * sigmoid(u_phase) to ensure phases lie in [0, 2π]
      - amplitudes = sigmoid(u_amp) to ensure amplitudes lie in [0, 1]
      
    It is assumed that the simulation and the cost function are compatible with PyTorch's automatic differentiation.
    """
    
    def __init__(self, cost_function: BaseCost, max_iter: int = 200, lr: float = 0.1) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.lr = lr

    def _get_coil_config_from_u(self, u: torch.Tensor) -> CoilConfig:
        # Split the 16-dimensional vector into phases and amplitudes (each of length 8).
        u_phase = u[:8]
        u_amp = u[8:]
        # Transform: map unconstrained parameters to desired ranges.
        phase = 2 * np.pi * torch.sigmoid(u_phase)
        amplitude = torch.sigmoid(u_amp)
        return CoilConfig(phase=phase, amplitude=amplitude)
    
    def optimize(self, simulation: Simulation):
        u = nn.Parameter(torch.randn(16, dtype=torch.float32))
        optimizer = Adam([u], lr=self.lr)

        best_cost = -np.inf if self.direction == "maximize" else np.inf
        best_coil_config = None

        pbar = trange(self.max_iter)
        for i in pbar:
            optimizer.zero_grad()
            # Map u to a valid coil configuration.
            coil_config = self._get_coil_config_from_u(u)
            # Run the simulation.
            # Assuming that the simulation function is adapted to work with torch tensors.
            simulation_data = simulation(coil_config)
            # Compute cost using the provided cost function.
            cost = self.cost_function(simulation_data)
            # For maximization problems, minimize the negative cost.
            loss = -cost if self.direction == "maximize" else cost
            loss.backward()
            optimizer.step()

            # For logging purposes, convert the cost to a Python float.
            current_cost = cost.item()
            if (self.direction == "minimize" and current_cost < best_cost) or \
               (self.direction == "maximize" and current_cost > best_cost):
                best_cost = current_cost
                # Create a new CoilConfig with cloned tensors.
                best_coil_config = CoilConfig(phase=coil_config.phase.clone(), 
                                              amplitude=coil_config.amplitude.clone())

            pbar.set_postfix_str(f"Loss: {loss.item():.4f} | Best cost: {best_cost:.4f}")

        # Return the coil configuration corresponding to the best cost found.
        return best_coil_config