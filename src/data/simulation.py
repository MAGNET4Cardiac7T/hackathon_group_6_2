import os
import h5py
import torch
import einops

from typing import Tuple
from .dataclasses import SimulationRawData, SimulationData, CoilConfig

class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5"):
        self.path = path
        self.coil_path = coil_path
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path
        
        def read_field() -> torch.Tensor:
            with h5py.File(self.path, 'r') as f:
                re_efield = torch.tensor(f["efield"]["re"][:], dtype=torch.float32)
                im_efield = torch.tensor(f["efield"]["im"][:], dtype=torch.float32)
                re_hfield = torch.tensor(f["hfield"]["re"][:], dtype=torch.float32)
                im_hfield = torch.tensor(f["hfield"]["im"][:], dtype=torch.float32)
                # First, stack the real and imaginary parts for the efield and hfield separately.
                efield = torch.stack([re_efield, im_efield], dim=0)
                hfield = torch.stack([re_hfield, im_hfield], dim=0)
                # Then, stack both fields along a new dimension.
                field = torch.stack([efield, hfield], dim=0)
            return field

        def read_physical_properties() -> torch.Tensor:
            with h5py.File(self.path, 'r') as f:
                physical_properties = torch.tensor(f["input"][:], dtype=torch.float32)
            return physical_properties
        
        def read_subject_mask() -> torch.Tensor:
            with h5py.File(self.path, 'r') as f:
                subject = torch.tensor(f["subject"][:], dtype=torch.bool)
            # Take maximum along the last axis to obtain a reduced mask.
            subject = torch.max(subject, dim=-1).values
            return subject
        
        def read_coil_mask() -> torch.Tensor:
            with h5py.File(self.coil_path, 'r') as f:
                coil = torch.tensor(f["masks"][:], dtype=torch.float32)
            return coil
        
        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask()
        )
        
        return simulation_raw_data
    
    def _shift_field(self, 
                     field: torch.Tensor, 
                     phase: torch.Tensor, 
                     amplitude: torch.Tensor) -> torch.Tensor:
        """
        Shift the field calculating field_shifted = field * amplitude * (e^(1j * phase)) 
        and summing over all coils.
        """
        # Create the real and imaginary parts of the phase factors.
        re_phase = torch.cos(phase) * amplitude
        im_phase = torch.sin(phase) * amplitude
        # Stack them: shape becomes (2, coils)
        coeffs_real = torch.stack((re_phase, -im_phase), dim=0)
        coeffs_im = torch.stack((im_phase, re_phase), dim=0)
        # Combine into a 2x2 tensor for each coil.
        coeffs = torch.stack((coeffs_real, coeffs_im), dim=0)
        # Expand coefficients along a new dimension 'hf' (for field type) so that
        # the shape fits field: repeat along hf dimension to match the two field types.
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        # Use einops.einsum to shift the field:
        # The equation multiplies the field by the phase/amplitude coefficients and sums over the coil dimension.
        field_shift = einops.einsum(field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        field_shifted = self._shift_field(self.simulation_raw_data.field, 
                                          coil_config.phase, 
                                          coil_config.amplitude)
        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        return self.phase_shift(coil_config)