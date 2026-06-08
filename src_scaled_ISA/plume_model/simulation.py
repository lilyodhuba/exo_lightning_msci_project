"""
Runner for the 1-D plume model experiment(s).


Examples
--------
>>> from plume_model import run_sim
>>> from constants import PhysicalConstants, SimulationParameters
>>> results = run_sim(
...     sim_params=SimulationParameters.from_yaml(),
...     const=PhysicalConstants.from_yaml()
... )
>>> print(f"Peak LFR: {results.extract_cube('flash_rate').data.max():.3f} fl s-1 km-2")
Peak LFR: 0.234 fl s-1 km-2
"""

import time

import iris
import paths
from constants import PhysicalConstants, SimulationParameters

from plume_model import run_sim

iris.FUTURE.save_split_attrs = True


def main():
    """Main simulation runner."""

    # run_label = (
    #     f"{sim_params.plume_base_temp:.0f}__{sim_params.temp_supercool:.0f}__"
    #     f"{sim_params.water_collision_efficiency:.1f}__{sim_params.ice_collision_efficiency:.1f}"
    # ).replace(".", "p")

    runs = ["default", "run01", "run02", "run03", "run04"]

    for run_label in runs:
        start_time = time.time()

        print(f"Running simulation: {run_label}")

        # Load the constants and simulation parameters from YAML files
        const = PhysicalConstants.from_yaml(
            paths.config / run_label / "physical_constants.yaml"
        )
        sim_params = SimulationParameters.from_yaml(
            paths.config / run_label / "simulation_parameters.yaml"
        )

        # Run the simulation
        result = run_sim(sim_params, const)

        for cube in result:
            cube.attributes["run_label"] = run_label
            cube.attributes["start_pressure"] = sim_params.start_pressure
            cube.attributes.update(sim_params.to_dict())
            cube.attributes.update(const.to_dict())

        # Save the result to a NetCDF file
        paths.plume_model_output.mkdir(parents=True, exist_ok=True)
        iris.save(
            result, paths.plume_model_output / f"plume_model_output_{run_label}.nc"
        )

        elapsed_time = time.time() - start_time
        print(f"Calculation time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
