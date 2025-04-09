import torch
from .dataclasses import SimulationData

class B1Calculator:
    """
    Class to calculate B1 field from simulation data using PyTorch.
    """

    def __call__(self, simulation_data: SimulationData) -> torch.Tensor:
        return self.calculate_b1_field(simulation_data)

    def calculate_b1_field(self, simulation_data: SimulationData) -> torch.Tensor:
        # Get the B-field tensor (assumed to be a list or tensor with indexing).
        # b_field is assumed to be such that:
        #   b_field[0] is b_x and b_field[1] is b_y.
        # We convert them to complex tensors.
        b_field = simulation_data.field[1]
        b_x = b_field[0].to(torch.complex64)
        b_y = b_field[1].to(torch.complex64)
        # Compute the complex B-field: b1_plus = b_x + i*b_y.
        b_field_complex = b_x + 1j * b_y
        # Now, form b1_plus = 0.5*(b_field_complex[0] + 1j*b_field_complex[1])
        # Here we assume that b_field_complex has at least two elements along its first dimension.
        b1_plus = 0.5 * (b_field_complex[0] + 1j * b_field_complex[1])
        return b1_plus
    

class SARCalculator:
    """
    Class to calculate SAR from simulation data using PyTorch.
    """

    def __call__(self, simulation_data: SimulationData) -> torch.Tensor:
        return self.calculate_sar(simulation_data)

    def calculate_sar(self, simulation_data: SimulationData) -> torch.Tensor:
        # e_field is assumed to be stored such that simulation_data.field[0] is the electric field.
        e_field = simulation_data.field[0]
        # Compute the squared magnitude of the electric field across dimensions 0 and 1.
        abs_efield_sq = torch.sum(e_field**2, dim=(0, 1))

        # Get the conductivity and density tensors.
        conductivity = simulation_data.properties[0]
        density = simulation_data.properties[2]

        pointwise_sar = conductivity * abs_efield_sq / density
        return pointwise_sar