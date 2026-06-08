"""Constants and configurable parameters for the lightning plume model."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import paths
import yaml


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in lightning plume model calculations."""

    gravity: float
    universal_gas_constant: float
    molar_mass_dry_air: float
    epsilon: float
    c_p: float
    latent_heat_v: float
    vacuum_perm: float
    e_charge: float
    rho_water: float
    rhoro: float
    drag_coef: float
    energy_per_flash: float
    mean_free_path_ion_coll: float
    temp_freeze: float
    pa_to_bar: float

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path] = paths.config_default / "physical_constants.yaml",
    ):
        """Load physical constants from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v["value"] for k, v in data["physical_constants"].items()})

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class SimulationParameters:
    """Configurable parameters for a single simulation run."""

    plume_base_temp: float
    base_humidity_fraction: float
    plume_base_radius: float
    temp_supercool: float
    water_collision_efficiency: float
    ice_collision_efficiency: float
    start_pressure: float
    start_upward_velocity: float
    pressure_step: float
    growth_time_step: float
    n_bins: int
    min_radius: float
    max_radius: float
    flash_rate_sampling: int
    dt: float
    project_name: str = "default_project"
    method_terminal_velocity: str = "aglyamov21"
    # method_terminal_velocity: str = "loftus21"

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path] = paths.config_default
        / "simulation_parameters.yaml",
    ):
        """Load simulation parameters from a YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(
            **{k: v["value"] for k, v in data["simulation_parameters"].items()},
            project_name=data["project"]["name"],
        )

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
