"""Production-grade model wrapper for cement-process KPI forecasting.

This module provides :class:`EnergyKPIModel`, a thin, self-describing wrapper
around a regression estimator (LightGBM by default, scikit-learn
``RandomForestRegressor`` as a fallback). The wrapper guarantees:

* deterministic feature ordering between training and inference,
* reproducible training metadata (timestamp, sample/feature counts, backend),
* explicit, fail-loud validation (no silent failures),
* simple ``save``/``load`` round-tripping of the *entire* wrapper via joblib.

The feature-engineering layer (:mod:`src.ml.features`) supplies the
``X: pd.DataFrame`` / ``y: pd.Series`` pair consumed here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Default LightGBM hyperparameters for KPI forecasting.
_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
}


def _build_default_estimator() -> tuple[Any, str]:
    """Build the default estimator, preferring LightGBM.

    Returns
    -------
    tuple[Any, str]
        The unfitted estimator and a human-readable backend identifier.
        Falls back to :class:`sklearn.ensemble.RandomForestRegressor` when
        LightGBM is not importable.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**_LGBM_PARAMS), "lightgbm.LGBMRegressor"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

        logger.warning(
            "LightGBM unavailable; falling back to RandomForestRegressor."
        )
        return RandomForestRegressor(random_state=42), "sklearn.RandomForestRegressor"


class EnergyKPIModel:
    """Self-describing regression wrapper for forecasting a single KPI/tag.

    Parameters
    ----------
    model:
        A pre-built (optionally fitted) estimator. When ``None``, a default
        estimator is constructed at :meth:`fit` time.
    feature_cols:
        Deterministic feature ordering. Normally populated by :meth:`fit`.
    target_name:
        Name of the predicted KPI/tag.
    model_name:
        Identifier for the backend model type.
    metadata:
        Free-form training metadata; populated by :meth:`fit`.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_cols: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        model_name: str = "lightgbm",
        metadata: Optional[dict] = None,
    ) -> None:
        self.model: Optional[Any] = model
        self.feature_cols: Optional[List[str]] = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.target_name: Optional[str] = target_name
        self.model_name: str = model_name
        self.metadata: dict = metadata or {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnergyKPIModel":
        """Fit the estimator and record reproducible training metadata.

        Parameters
        ----------
        X:
            Feature matrix. Column order is captured verbatim and enforced at
            inference time.
        y:
            Target vector aligned to ``X``.

        Returns
        -------
        EnergyKPIModel
            ``self``, to allow fluent chaining.

        Raises
        ------
        TypeError
            If ``X`` is not a DataFrame or ``y`` is not a Series.
        ValueError
            If ``X`` or ``y`` is empty, or their lengths differ.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)!r}.")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)!r}.")
        if X.empty:
            raise ValueError("X must be non-empty.")
        if y.empty:
            raise ValueError("y must be non-empty.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} rows vs {len(y)} targets."
            )

        self.feature_cols = list(X.columns)

        if self.model is not None:
            estimator: Any = self.model
            backend = f"{type(estimator).__module__}.{type(estimator).__name__}"
        else:
            estimator, backend = _build_default_estimator()

        estimator.fit(X[self.feature_cols], y)
        self.model = estimator
        self.model_name = backend.split(".")[0]

        if self.target_name is None and y.name is not None:
            self.target_name = str(y.name)

        self.metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "feature_names": list(self.feature_cols),
            "model_backend": backend,
            "target_name": self.target_name,
        }
        logger.info(
            "Fitted %s on %d samples x %d features (target=%s)",
            backend,
            self.metadata["n_samples"],
            self.metadata["n_features"],
            self.target_name,
        )
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for ``X``, enforcing training feature order.

        Parameters
        ----------
        X:
            Feature matrix containing (at least) every training feature.

        Returns
        -------
        numpy.ndarray
            Model predictions.

        Raises
        ------
        TypeError
            If ``X`` is not a DataFrame.
        ValueError
            If the model is not fitted, or required feature columns are
            missing from ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)!r}.")
        if not self.is_fitted:
            raise ValueError("Model is not fitted; call fit() before predict().")

        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required feature column(s) at inference: {missing}."
            )

        ordered = X[self.feature_cols]
        return np.asarray(self.model.predict(ordered))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist the full wrapper (model + metadata) to ``path`` via joblib.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created as needed.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)
        logger.info("Saved EnergyKPIModel to %s", out)

    @classmethod
    def load(cls, path: str) -> "EnergyKPIModel":
        """Load a persisted wrapper from ``path``.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save`.

        Returns
        -------
        EnergyKPIModel
            The deserialized wrapper.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        TypeError
            If the persisted object is not an :class:`EnergyKPIModel`.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Model artifact not found: {src}")

        obj = joblib.load(src)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Persisted object is {type(obj)!r}, expected {cls.__name__}."
            )
        logger.info("Loaded EnergyKPIModel from %s", src)
        return obj

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        """Whether the wrapper holds a fitted model and a feature ordering."""
        return self.model is not None and bool(self.feature_cols)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a descending-sorted DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature`` and ``importance``, sorted by importance
            descending.

        Raises
        ------
        ValueError
            If the model is not fitted, or the underlying estimator does not
            expose ``feature_importances_``.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted; nothing to report.")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError(
                f"Estimator {type(self.model).__name__} exposes no "
                "feature_importances_."
            )

        importances = np.asarray(self.model.feature_importances_, dtype=float)
        frame = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        )
        return (
            frame.sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
