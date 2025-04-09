import torch
from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

class B1HomogeneityCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()

    def calculate_cost(self, simulation_data: SimulationData) -> torch.Tensor:
        # Compute the B1 field as a torch tensor.
        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject  # Expected to be a boolean mask or tensor for torch indexing

        # Compute the absolute value using torch.
        b1_field_abs = torch.abs(b1_field)
        # Select the voxels corresponding to the subject.
        b1_field_subject_voxels = b1_field_abs[subject]
        
        # Compute the cost as the ratio of the mean to the standard deviation.
        return torch.mean(b1_field_subject_voxels) / torch.std(b1_field_subject_voxels)
