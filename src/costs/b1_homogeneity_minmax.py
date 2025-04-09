import torch
from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

class B1HomogeneityMinMaxCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()

    def calculate_cost(self, simulation_data: SimulationData) -> torch.Tensor:
        # Compute the B1 field as a torch tensor.
        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject  # Assumed to be a boolean mask or index tensor for torch indexing

        # Compute the absolute value using torch operations.
        b1_field_abs = torch.abs(b1_field)
        # Select the subject voxels from the B1 field.
        b1_field_subject_voxels = b1_field_abs[subject]

        # Calculate the numerator and denominator using torch's mean, max, and min.
        numerator = torch.mean(b1_field_subject_voxels)
        denominator = torch.max(b1_field_subject_voxels) - torch.min(b1_field_subject_voxels)

        # Return the ratio as the cost.
        cost = numerator / denominator
        return cost