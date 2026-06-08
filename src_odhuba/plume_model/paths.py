"""Common paths useful for manipulating datasets and generating figures."""

from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src_odhuba = root / "src_odhuba"

# Absolute path to the `src/data` folder (contains datasets)
data = src_odhuba / "data"

# Absolute path to the `src/plume_model_output` folder (contains model output)
plume_model_output = data / "plume_model_output"

# Absolute path to the `src/plume_model` folder (contains model code)
model = src_odhuba / "plume_model"

# Absolute path to the `src/config` folder (contains YAML config files)
config = model / "config"

# Default config path
config_default = model / "config" / "default"

# Absolute path to the `src/figures` folder (contains figure output)
figures = src_odhuba / "figures"