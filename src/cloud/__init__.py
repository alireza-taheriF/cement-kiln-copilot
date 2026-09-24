"""Cloud training launchers.

Local training (``python -m src.ml.train``) stays the default. These modules
only print a job spec (``--dry-run``) or submit that same command when cloud
credentials are present (``--submit``).
"""

from src.cloud.training_command import build_training_command

__all__ = ["build_training_command"]
