"""Naive persistence and linear-regression baselines on a chronological split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data_ingest.loader import save_signals_to_db
from src.db.models import get_engine, get_session_factory, init_db
from src.ml.baselines import (
    evaluate_naive_baselines,
    fit_linear_regression,
    last_observed_target,
    persistence_predict,
)
from src.ml.evaluate import regression_metrics, train_valid_split_time_series
from src.ml.features import make_supervised_dataset, pivot_signals
from src.ml.train import train_energy_model


def _tiny_signals(periods: int = 16) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="min")
    temp = pd.DataFrame(
        {
            "timestamp": index,
            "tag_name": "KILN_ZONE1_TEMP",
            "value": np.arange(periods, dtype=float) + 10.0,
        }
    )
    fuel = pd.DataFrame(
        {
            "timestamp": index,
            "tag_name": "KILN_FUEL_FLOW",
            "value": np.arange(periods, dtype=float) * 0.5,
        }
    )
    return pd.concat([temp, fuel], ignore_index=True)


def test_persistence_is_deterministic_and_uses_forecast_time_value():
    observed = pd.Series([1.5, 2.5, 3.5], name="KILN_ZONE1_TEMP")
    first = persistence_predict(observed)
    second = persistence_predict(observed)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.array([1.5, 2.5, 3.5]))
    # A future label must not be what persistence emits.
    future = np.array([90.0, 91.0, 92.0])
    assert not np.array_equal(first, future)


def test_last_observed_target_is_the_origin_not_the_future_label():
    signals = _tiny_signals(periods=16)
    horizon = 3
    X, y = make_supervised_dataset(
        signals,
        target_tag="KILN_ZONE1_TEMP",
        input_tags=["KILN_ZONE1_TEMP", "KILN_FUEL_FLOW"],
        lags=[1],
        rolling_windows=[3],
        horizon=horizon,
    )
    observed = last_observed_target(signals, "KILN_ZONE1_TEMP", X.index)
    wide = pivot_signals(signals)
    assert observed.index.equals(X.index)
    assert np.allclose(
        observed.to_numpy(),
        wide.loc[X.index, "KILN_ZONE1_TEMP"].to_numpy(),
    )
    assert not np.allclose(observed.to_numpy(), y.to_numpy())


def test_persistence_score_ignores_future_labels():
    index = pd.RangeIndex(6)
    X = pd.DataFrame({"f": np.arange(6, dtype=float)}, index=index)
    y = pd.Series(np.arange(6, dtype=float) * 10.0 + 100.0, index=index, name="y")
    observed = pd.Series([1.0, 1.0, 2.5, 2.5, 4.0, 4.0], index=index)
    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X, y, valid_fraction=0.34
    )
    report_a = evaluate_naive_baselines(
        X_train,
        y_train,
        X_valid,
        y_valid,
        observed,
        model_valid_rmse=1.25,
    )
    report_b = evaluate_naive_baselines(
        X_train,
        y_train,
        X_valid,
        y_valid,
        observed,
        model_valid_rmse=1.25,
    )
    assert report_a == report_b

    expected_pred = persistence_predict(observed.loc[X_valid.index])
    expected = regression_metrics(y_valid, expected_pred)
    assert report_a["persistence"]["rmse"] == expected["rmse"]
    assert report_a["persistence"]["mae"] == expected["mae"]
    assert report_a["persistence"]["n_samples"] == len(y_valid)
    assert report_a["n_samples"] == len(y_valid)
    assert report_a["model_valid_rmse"] == 1.25
    assert not np.allclose(expected_pred, y_valid.to_numpy())
    assert report_a["persistence"]["rmse"] != 0.0


def _supervised_without_target_as_feature(signals: pd.DataFrame):
    """Features from fuel only, so a missing target origin can stay in X."""
    return make_supervised_dataset(
        signals,
        target_tag="KILN_ZONE1_TEMP",
        input_tags=["KILN_FUEL_FLOW"],
        lags=[1],
        rolling_windows=[3],
        horizon=2,
    )


def test_missing_training_origin_observation_still_scores_validation(tmp_path):
    """A hole at a training origin must not block validation persistence."""
    signals = _tiny_signals(periods=40)
    X, y = _supervised_without_target_as_feature(signals)
    X_train, X_valid, _, _ = train_valid_split_time_series(X, y, valid_fraction=0.2)
    train_origin = X_train.index[5]
    assert train_origin not in X_valid.index

    broken = signals.loc[
        ~(
            (signals["tag_name"] == "KILN_ZONE1_TEMP")
            & (signals["timestamp"] == train_origin)
        )
    ].copy()
    X_broken, y_broken = _supervised_without_target_as_feature(broken)
    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X_broken, y_broken, valid_fraction=0.2
    )
    assert train_origin in X_train.index
    assert train_origin not in X_valid.index

    with pytest.raises(ValueError, match="future"):
        last_observed_target(broken, "KILN_ZONE1_TEMP", X_broken.index)

    observed_valid = last_observed_target(broken, "KILN_ZONE1_TEMP", X_valid.index)
    assert observed_valid.index.equals(X_valid.index)
    assert not observed_valid.isna().any()
    report = evaluate_naive_baselines(
        X_train,
        y_train,
        X_valid,
        y_valid,
        observed_valid,
        model_valid_rmse=1.5,
    )
    assert report["n_samples"] == len(X_valid)
    assert report["persistence"]["n_samples"] == len(X_valid)
    assert report["linear_regression"]["n_samples"] == len(X_valid)
    assert np.isfinite(report["persistence"]["rmse"])
    assert report["model_valid_rmse"] == 1.5

    valid_origin = X_valid.index[0]
    missing_valid = signals.loc[
        ~(
            (signals["tag_name"] == "KILN_ZONE1_TEMP")
            & (signals["timestamp"] == valid_origin)
        )
    ].copy()
    with pytest.raises(ValueError, match="future"):
        last_observed_target(missing_valid, "KILN_ZONE1_TEMP", X_valid.index)

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    session = get_session_factory(engine)()
    try:
        save_signals_to_db(broken, session)
        summary = train_energy_model(
            session,
            target_tag="KILN_ZONE1_TEMP",
            input_tags=["KILN_FUEL_FLOW"],
            lags=[1],
            rolling_windows=[3],
            horizon=2,
            model_output_path=str(tmp_path / "model.joblib"),
        )
    finally:
        session.close()

    comparison = summary["baseline_comparison"]
    assert comparison["n_samples"] == summary["n_valid_samples"]
    assert comparison["n_samples"] > 0
    assert comparison["model_valid_rmse"] == summary["valid_metrics"]["rmse"]
    assert np.isfinite(comparison["persistence"]["rmse"])
    assert np.isfinite(comparison["linear_regression"]["rmse"])


def test_linear_regression_fits_training_rows_only():
    rng = np.random.default_rng(0)
    n = 30
    x = np.linspace(0.0, 5.0, n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    y_values = np.empty(n)
    y_values[:24] = 3.0 * x[:24] + 1.0
    y_values[24:] = -20.0 * x[24:] + 100.0
    y = pd.Series(y_values, name="kpi")
    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X, y, valid_fraction=0.2
    )
    assert len(X_train) == 24
    assert list(X_valid.index) == list(range(24, 30))

    observed = pd.Series(np.arange(n, dtype=float), index=X.index)
    report = evaluate_naive_baselines(
        X_train,
        y_train,
        X_valid,
        y_valid,
        observed,
        model_valid_rmse=0.5,
    )

    train_only = fit_linear_regression(X_train, y_train)
    leaked = LinearRegression().fit(
        pd.concat([X_train, X_valid]),
        pd.concat([y_train, y_valid]),
    )
    pred_train = np.asarray(train_only.predict(X_valid), dtype=float)
    pred_leaked = np.asarray(leaked.predict(X_valid), dtype=float)
    assert not np.allclose(pred_train, pred_leaked)

    expected = regression_metrics(y_valid, pred_train)
    leaked_metrics = regression_metrics(y_valid, pred_leaked)
    assert report["linear_regression"]["rmse"] == pytest.approx(expected["rmse"])
    assert report["linear_regression"]["mae"] == pytest.approx(expected["mae"])
    assert report["linear_regression"]["rmse"] != pytest.approx(leaked_metrics["rmse"])
    assert report["linear_regression"]["n_samples"] == len(y_valid)
    assert report["n_samples"] == len(y_valid)
    assert report["model_valid_rmse"] == 0.5
    # The public fit helper must not have seen the validation targets.
    assert train_only.n_features_in_ == X_train.shape[1]
    assert len(train_only.coef_) == X_train.shape[1]


def test_evaluate_naive_baselines_rejects_a_shuffled_split():
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    y = pd.Series(np.arange(10, dtype=float))
    X_train, X_valid, y_train, y_valid = train_valid_split_time_series(
        X, y, valid_fraction=0.2
    )
    observed = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="shuffle"):
        evaluate_naive_baselines(
            X_valid,
            y_valid,
            X_train,
            y_train,
            observed,
            model_valid_rmse=1.0,
        )
