"""Shared training command for local runs and cloud job specs.

Local training stays the default. Cloud modules only wrap this argv.
"""

from __future__ import annotations

DEFAULT_TRAINING_CONFIG = "config/model_config.yaml"


def build_training_command(
    config_path: str = DEFAULT_TRAINING_CONFIG,
) -> list[str]:
    """Return the argv used to train a kiln KPI model.

    The default command is::

        python -m src.ml.train --config config/model_config.yaml
    """
    return ["python", "-m", "src.ml.train", "--config", config_path]
