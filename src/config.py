"""
Configuration loader for Local Drift-Adapter experiments.

Loads base.yaml, merges experiment-specific overrides on top,
and supports CLI overrides like --adapter.type=affine.
"""

import copy
import yaml
from pathlib import Path


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, returning a new dict.

    - Dict values are merged recursively.
    - All other values in overlay replace those in base.
    - Keys in overlay that don't exist in base are added.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(args: list[str]) -> dict:
    """Parse CLI override arguments into a nested dict.

    Each argument should be in the form --dotted.key=value, e.g.:
        ["--adapter.type=affine", "--clustering.n_clusters=16"]

    Returns:
        Nested dict, e.g. {"adapter": {"type": "affine"}, "clustering": {"n_clusters": 16}}
    """
    result: dict = {}
    for arg in args:
        if not arg.startswith("--"):
            continue
        arg = arg[2:]  # strip leading --
        if "=" not in arg:
            # Treat bare flags as boolean true (e.g. --adapter.fit_scaling)
            key_str, raw_value = arg, "true"
        else:
            key_str, raw_value = arg.split("=", 1)

        value = _cast_value(raw_value)

        # Build nested dict from dotted key
        keys = key_str.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def _cast_value(raw: str):
    """Cast a string value to the appropriate Python type."""
    # None / null
    if raw.lower() in ("null", "none"):
        return None
    # Booleans
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    # Int
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    # Lists (bracket syntax, e.g. [1,5,10])
    if raw.startswith("[") and raw.endswith("]"):
        items = raw[1:-1].split(",")
        return [_cast_value(item.strip()) for item in items if item.strip()]
    # String
    return raw


def load_config(
    experiment_config: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load configuration by merging base, experiment, and manual overrides.

    Args:
        experiment_config: Name of an experiment YAML file (e.g. "local_drift_aware"
            or "local_drift_aware.yaml"), or an absolute/relative path to one.
            If None, only base.yaml is loaded.
        overrides: Additional dict of overrides to merge on top.

    Returns:
        Fully merged configuration dict.
    """
    # 1. Load base config
    base_path = CONFIGS_DIR / "base.yaml"
    with open(base_path) as f:
        config = yaml.safe_load(f)

    # 2. Merge experiment config if provided
    if experiment_config is not None:
        exp_path = _resolve_config_path(experiment_config)
        with open(exp_path) as f:
            exp = yaml.safe_load(f)
        if exp:
            config = deep_merge(config, exp)

    # 3. Merge manual overrides
    if overrides:
        config = deep_merge(config, overrides)

    return config


def _resolve_config_path(name: str) -> Path:
    """Resolve an experiment config name to a file path.

    Accepts:
        - A bare name like "local_drift_aware" (looks in configs/)
        - A name with .yaml extension like "local_drift_aware.yaml"
        - An absolute or relative file path
    """
    path = Path(name)
    # If it's already an absolute path or an existing relative path, use it directly
    if path.is_absolute() or path.exists():
        return path
    # Otherwise look in the configs directory
    if not name.endswith(".yaml"):
        name = name + ".yaml"
    candidate = CONFIGS_DIR / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Experiment config '{name}' not found. "
        f"Looked in {CONFIGS_DIR} and current directory."
    )
