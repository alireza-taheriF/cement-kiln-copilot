"""Delay estimation, conformal intervals, delay-aware training, feature reconstruction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.domain import tags as T
from src.ml.delays import DelayEstimate, estimate_delay, lags_from_delays
from src.ml.inference import features_from_wide, min_window_rows, parse_feature_spec
from src.ml.pipeline import fit_kpi_model, prepare_wide
from src.ml.uncertainty import calibrate, empirical_coverage, interval


def test_estimate_delay_recovers_a_known_dead_time():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(size=2000)), name="x")
    y = x.shift(7).fillna(0) + rng.normal(scale=0.1, size=2000)
    est = estimate_delay(x, y, max_delay=20, diff_rows=1)
    assert est.delay_rows == 7
    assert est.correlation > 0.9
    assert est.is_reliable


def test_estimate_delay_validates_arguments():
    s = pd.Series(range(10), dtype=float)
    with pytest.raises(ValueError):
        estimate_delay(s, s, max_delay=1, min_delay=3)
    with pytest.raises(ValueError):
        estimate_delay(s, s, max_delay=3, diff_rows=0)


def test_lags_from_delays_adds_only_reliable_delays():
    delays = {
        "fuel": DelayEstimate("fuel", 9, 0.4, 500),
        "noise": DelayEstimate("noise", 15, 0.02, 500),
    }
    lags = lags_from_delays(delays, horizon=6, base_lags=[1])
    assert lags["fuel"] == [1, 3]
    assert lags["noise"] == [1]


def test_conformal_interval_has_nominal_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    pred = np.zeros(2000)
    cal = calibrate(y[:1000], pred[:1000], alpha=0.1)
    lo, hi = interval(pred[1000:], cal)
    assert empirical_coverage(y[1000:], lo, hi) == pytest.approx(0.9, abs=0.03)


def test_conformal_with_too_few_points_is_unbounded():
    assert math.isinf(calibrate([1.0, 2.0], [1.0, 1.0], alpha=0.1).half_width)
    with pytest.raises(ValueError):
        calibrate([], [], alpha=0.1)


def test_parse_feature_spec_and_window():
    cols = ["A_lag_0", "A_lag_3", "B_lag_1", "A_roll_mean_6"]
    lags, roll_tags, windows = parse_feature_spec(cols)
    assert lags == {"A": {0, 3}, "B": {1}}
    assert roll_tags == {"A"} and windows == {6}
    with pytest.raises(ValueError):
        parse_feature_spec(["weird"])


def test_auto_lag_model_is_calibrated_and_rebuilds_features(sim, constraint_models):
    model = constraint_models["bz_temp"]
    assert model.metadata["conformal"]["half_width"] > 0
    assert model.metadata["horizon_rows"] == 15
    assert "delays" in model.metadata
    # The derived KPI is rebuilt from raw tags at inference time.
    raw = sim.wide.drop(columns=[T.SPECIFIC_HEAT_KCAL_KG, T.GAS_HEAT_SHARE, T.HEAT_INPUT_GCALH])
    window = raw.iloc[3100 : 3100 + 3 * min_window_rows(model)]
    X = features_from_wide(window, model)
    assert list(X.columns) == model.feature_cols
    pred, lo, hi = model.predict_interval(X)
    assert np.all(lo < pred) and np.all(pred < hi)


def test_burning_zone_model_is_honest_out_of_sample(sim, constraint_models):
    """75 min ahead the kiln is barely predictable beyond persistence (the
    future fuel moves are unknown). The model must not be meaningfully worse
    than "BZT stays where it is", and its 90 % interval must cover close to
    90 % of unseen data: the advisor relies on the interval, not the point.
    """
    model = constraint_models["bz_temp"]
    test = sim.wide.iloc[3000:]
    X = features_from_wide(test, model)
    horizon = model.metadata["horizon_rows"]
    future = test[T.BZ_TEMP_C].shift(-horizon).reindex(X.index)
    keep = future.notna()
    pred, lower, upper = model.predict_interval(X[keep])
    rmse_model = float(np.sqrt(np.mean((pred - future[keep]) ** 2)))
    persistence = test[T.BZ_TEMP_C].reindex(X.index)[keep]
    rmse_persistence = float(np.sqrt(np.mean((persistence - future[keep]) ** 2)))
    assert rmse_model <= 1.1 * rmse_persistence
    assert empirical_coverage(future[keep], lower, upper) >= 0.8


def test_config_estimator_choice(sim):
    from src.ml.pipeline import build_estimator

    assert build_estimator(None) is None
    assert build_estimator("ridge") is not None
    with pytest.raises(ValueError):
        build_estimator("xgboost")


def test_fit_rejects_bad_lag_spec(sim):
    wide = prepare_wide(sim.wide.iloc[:500], [T.BZ_TEMP_C, T.ID_FAN_PCT])
    with pytest.raises(ValueError):
        fit_kpi_model(wide, T.BZ_TEMP_C, [T.BZ_TEMP_C, T.ID_FAN_PCT], "sometimes", [3], 2)


def test_prepare_wide_reports_missing_tags(sim):
    with pytest.raises(ValueError, match="not found"):
        prepare_wide(sim.wide, ["NOPE"])
