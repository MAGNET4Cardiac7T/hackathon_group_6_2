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
    
    def crop_to_mask(self, mask):
        # Find nonzero (True) indices
        nonzero = mask.nonzero(as_tuple=False)

        if nonzero.numel() == 0:
            return None  # or tensor.new_empty(...)

        # Find bounding box across all dimensions
        mins = nonzero.min(dim=0).values
        maxs = nonzero.max(dim=0).values + 1  # +1 to make slicing inclusive

        # Slice for each dimension
        slices = tuple(slice(mins[i].item(), maxs[i].item()) for i in range(mask.dim()))

        return slices
        
    def _shift_field(self, 
                     field: torch.Tensor, 
                     phase: torch.Tensor, 
                     amplitude: torch.Tensor) -> torch.Tensor:
        """
        Shift the field calculating field_shifted = field * amplitude * (e^(1j * phase)) 
        and summing over all coils.
        """
        re_phase = torch.cos(phase) * amplitude
        im_phase = torch.sin(phase) * amplitude
        coeffs_real = torch.stack((re_phase, -im_phase), axis=0)
        coeffs_im = torch.stack((im_phase, re_phase), axis=0)
        coeffs = torch.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        
        original_shape = field.shape  # For later use
        slices = self.crop_to_mask(self.simulation_raw_data.subject)
        masked_field = field[:, :, :,slices[0],slices[1],slices[2], :]  # shape: (2, 2, 3, N, 8), where N = number of True voxels
        
        field_shift = einops.einsum(masked_field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        
        restored = torch.zeros(
            (original_shape[0],  # hf
            original_shape[1],  # reimout (maybe different from input)
            original_shape[2],  # fieldxyz
            original_shape[3],     # full spatial X
            original_shape[4],     # full spatial Y
            original_shape[5],     # full spatial Z
            ),
            dtype=field_shift.dtype,
            device=field_shift.device
        )

        # Insert the processed region back into the original space
        restored[:, :, :, slices[0], slices[1], slices[2]] = field_shift
        return restored

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