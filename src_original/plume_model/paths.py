"""Common paths useful for manipulating datasets and generating figures."""

from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src_original = root / "src_original"

# Absolute path to the `src/data` folder (contains datasets)
data = src_original / "data"

# Absolute path to the `src/plume_model_output` folder (contains model output)
plume_model_output = data / "plume_model_output"

# Absolute path to the `src/ind_plume_model_output` folder (contains model output for individual plots)
ind_plume_model_output = data / "ind_plume_model_outputs"

# Absolute path to the `src/plume_model` folder (contains model code)
model = src_original / "plume_model"

# Absolute path to the `src/config` folder (contains YAML config files)
config = model / "config"

# Default config path
config_default = model / "config" / "default"

# Absolute path to the `src/figures` folder (contains figure output)
figures = src_original / "figures"

# Absolute path to the `src/ind_figures` folder (contains figure output in individual plots)
ind_figures = src_original / "ind_figures"