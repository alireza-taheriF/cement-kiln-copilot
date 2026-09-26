"""Machine learning package.

Contains feature engineering, model definitions, training, evaluation,
and recommendation logic for kiln/mill optimization.
"""

from .baselines import (
    evaluate_naive_baselines,
    fit_linear_regression,
    last_observed_target,
    persistence_predict,
)
from .features import (
    create_lag_features,
    create_rolling_features,
    create_target,
    make_supervised_dataset,
    pivot_signals,
)
from .evaluate import (
    evaluate_regressor,
    regression_metrics,
    train_valid_split_time_series,
)
from .models import EnergyKPIModel
from .recommend import (
    apply_candidate_to_signals,
    generate_candidates,
    rank_candidates,
    score_candidate,
)

__all__ = [
    "last_observed_target",
    "persistence_predict",
    "fit_linear_regression",
    "evaluate_naive_baselines",
    "pivot_signals",
    "create_lag_features",
    "create_rolling_features",
    "create_target",
    "make_supervised_dataset",
    "EnergyKPIModel",
    "regression_metrics",
    "evaluate_regressor",
    "train_valid_split_time_series",
    "generate_candidates",
    "apply_candidate_to_signals",
    "score_candidate",
    "rank_candidates",
]
