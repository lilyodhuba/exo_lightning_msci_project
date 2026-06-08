from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import iris
import matplotlib.pyplot as plt
import paths
from constants import PhysicalConstants, SimulationParameters
from iris.cube import CubeList


def plot_comparison(
    results: dict[CubeList],
    output_dir: Union[str, Path],
    sim_params: SimulationParameters,
    const: PhysicalConstants,
):
    """Create comparison plots for a series of simulations."""

    @dataclass
    class _PlotConfig:
        """Configuration for a single plot."""

        ylabel: str
        title: str
        units: str

    plot_configs = {
        "velocity": _PlotConfig(
            ylabel="Vertical velocity",
            title="Vertical Plume Velocity",
            units="m s-1",
        ),
        "plume_temp": _PlotConfig(
            ylabel="Temperature", title="Plume Temperature", units="K"
        ),
        "env_temp": _PlotConfig(
            ylabel="Temperature", title="Environment Temperature", units="K"
        ),
        "temp_diff": _PlotConfig(
            ylabel="Temperature difference",
            title="Plume-Environment Temperature Difference",
            units="K",
        ),
        "plume_radius": _PlotConfig(ylabel="Radius", title="Plume Radius", units="m"),
        "flash_rate": _PlotConfig(
            ylabel="Flash rate",
            title="Lightning Flash Rate",
            units="flashes s-1 km-2",
        ),
    }

    # Create figure with mosaic layout using plot_configs keys
    mosaic = [list(plot_configs.keys())[:3], list(plot_configs.keys())[3:]]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axes = fig.subplot_mosaic(mosaic)

    # Plot each variable
    handles, labels = None, None
    for name, config in plot_configs.items():
        ax = axes[name]

        for run_label, cube_list in results.items():
            if name == "temp_diff":
                # Plume temp - Env temp
                x = cube_list.extract_cube("plume_temp") - cube_list.extract_cube(
                    "env_temp"
                )
                
                pressure = cube_list.extract_cube("plume_temp").coord("air_pressure").points
                initial_pressure = float(cube_list[0].attributes["start_pressure"])                
                y = pressure / float(initial_pressure)
            
            else:
                x = cube_list.extract_cube(name)
                initial_pressure = float(cube_list[0].attributes["start_pressure"])                
                y = (x.coord("air_pressure").points) / float(initial_pressure)
            line = ax.plot(x.data, y, label=run_label, linewidth=1.5)

            # Capture handles and labels from the first subplot
            if handles is None:
                handles, labels = [], []
            if name == list(plot_configs.keys())[0]:
                handles.extend(line)
                labels.append(run_label)

        ax.set_ylabel(r"$\sigma$")
        ax.invert_yaxis()
        # ax.set_ylim(1e5 * const.pa_to_bar, 0)
        ax.set_xlabel(f"{config.ylabel} [{config.units}]")
        ax.set_title(config.title)
        ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    fig.suptitle(
        "1D Model of Convective Plume for Earth-like Exoplanet (with ISA)",
        fontsize="x-large",
        fontweight="bold",
        y=1.075,
    )

    filename = f"{sim_params.project_name}_new.png"
    fig.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


const = PhysicalConstants.from_yaml()
sim_params = SimulationParameters.from_yaml()

results = {}
results["0p25 bar"] = iris.load(
    paths.plume_model_output / "plume_model_output_default.nc"
)
results["0p5 bar"] = iris.load(
    paths.plume_model_output / "plume_model_output_run01.nc"
)
results["1 bar"] = iris.load(
    paths.plume_model_output / "plume_model_output_run02.nc"
)
results["2 bar"] = iris.load(
    paths.plume_model_output / "plume_model_output_run03.nc"
)
results["4 bar"] = iris.load(
    paths.plume_model_output / "plume_model_output_run04.nc"
)



print("Generating plots...")
plot_comparison(results, paths.figures, sim_params, const)
print("Done.")
