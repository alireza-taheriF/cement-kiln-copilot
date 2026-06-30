"""Tests for the ML pipeline (features, models, evaluation, recommend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ml.evaluate import (
    evaluate_regressor,
    regression_metrics,
    train_valid_split_time_series,
)
from src.ml.features import (
    create_lag_features,
    create_rolling_features,
    create_target,
    make_supervised_dataset,
    pivot_signals,
)
from src.ml.models import EnergyKPIModel
from src.ml.train import (
    SIGNAL_COLUMNS,
    load_signals_from_db,
    load_training_config,
    train_energy_model,
)
from src.data_ingest.loader import save_signals_to_db
from src.db.models import get_engine, get_session_factory, init_db


def _long_signals(periods: int = 12) -> pd.DataFrame:
    """Two-tag long-format signal frame for feature-engineering tests."""
    index = pd.date_range("2026-01-01 10:00", periods=periods, freq="min")
    temp = pd.DataFrame(
        {
            "timestamp": index,
            "tag_name": "KILN_ZONE1_TEMP",
            "value": np.arange(periods, dtype=float) + 1380.0,
        }
    )
    fuel = pd.DataFrame(
        {
            "timestamp": index,
            "tag_name": "KILN_FUEL_FLOW",
            "value": np.arange(periods, dtype=float) + 42.0,
        }
    )
    return pd.concat([temp, fuel], ignore_index=True)


def test_pivot_signals_wide_format_sorted_and_keeps_last():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:01",
                    "2026-01-01 10:00",
                    "2026-01-01 10:00",  # duplicate (timestamp, tag) -> keep last
                ]
            ),
            "tag_name": ["KILN_ZONE1_TEMP", "KILN_ZONE1_TEMP", "KILN_ZONE1_TEMP"],
            "value": [1381.0, 1379.0, 1380.0],
        }
    )
    wide = pivot_signals(df)
    assert wide.index.is_monotonic_increasing
    assert list(wide.columns) == ["KILN_ZONE1_TEMP"]
    # Duplicate first timestamp kept the last value (1380.0, not 1379.0).
    assert wide.iloc[0, 0] == 1380.0


def test_pivot_signals_raises_on_missing_columns():
    with pytest.raises(ValueError):
        pivot_signals(pd.DataFrame({"timestamp": [], "value": []}))


def test_create_lag_features_naming_and_values():
    wide = pivot_signals(_long_signals())
    lags = create_lag_features(wide, input_tags=["KILN_ZONE1_TEMP"], lags=[1, 5])
    assert list(lags.columns) == ["KILN_ZONE1_TEMP_lag_1", "KILN_ZONE1_TEMP_lag_5"]
    # lag_1 at position 1 equals the raw value at position 0.
    assert lags["KILN_ZONE1_TEMP_lag_1"].iloc[1] == wide["KILN_ZONE1_TEMP"].iloc[0]


def test_create_lag_features_raises_on_missing_tag():
    wide = pivot_signals(_long_signals())
    with pytest.raises(ValueError):
        create_lag_features(wide, input_tags=["DOES_NOT_EXIST"], lags=[1])


def test_create_rolling_features_emits_all_aggregations():
    wide = pivot_signals(_long_signals())
    roll = create_rolling_features(wide, input_tags=["KILN_FUEL_FLOW"], windows=[3])
    expected = [
        "KILN_FUEL_FLOW_roll_mean_3",
        "KILN_FUEL_FLOW_roll_std_3",
        "KILN_FUEL_FLOW_roll_min_3",
        "KILN_FUEL_FLOW_roll_max_3",
    ]
    assert list(roll.columns) == expected
    # Third row (index 2) is the first full window of [42, 43, 44].
    assert roll["KILN_FUEL_FLOW_roll_mean_3"].iloc[2] == 43.0
    assert roll["KILN_FUEL_FLOW_roll_min_3"].iloc[2] == 42.0
    assert roll["KILN_FUEL_FLOW_roll_max_3"].iloc[2] == 44.0


def test_create_target_shifts_backward_by_horizon():
    wide = pivot_signals(_long_signals())
    y = create_target(wide, target_tag="KILN_ZONE1_TEMP", horizon=2)
    assert y.name == "KILN_ZONE1_TEMP_h2"
    # y[t] == target[t + 2].
    assert y.iloc[0] == wide["KILN_ZONE1_TEMP"].iloc[2]
    # Last `horizon` rows have no future value.
    assert y.iloc[-2:].isna().all()


def test_create_target_raises_on_bad_inputs():
    wide = pivot_signals(_long_signals())
    with pytest.raises(ValueError):
        create_target(wide, target_tag="MISSING", horizon=1)
    with pytest.raises(ValueError):
        create_target(wide, target_tag="KILN_ZONE1_TEMP", horizon=0)


def test_make_supervised_dataset_aligns_and_drops_nans():
    signals = _long_signals(periods=12)
    X, y = make_supervised_dataset(
        signals,
        target_tag="KILN_ZONE1_TEMP",
        input_tags=["KILN_ZONE1_TEMP", "KILN_FUEL_FLOW"],
        lags=[1, 2],
        rolling_windows=[3],
        horizon=2,
    )
    # X and y are aligned with no missing values.
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
    # Deterministic column ordering: all lag columns precede rolling columns.
    lag_cols = [c for c in X.columns if "_lag_" in c]
    roll_cols = [c for c in X.columns if "_roll_" in c]
    assert list(X.columns) == lag_cols + roll_cols
    assert X.index.equals(y.index)


def test_make_supervised_dataset_raises_on_missing_tags():
    signals = _long_signals()
    with pytest.raises(ValueError):
        make_supervised_dataset(
            signals,
            target_tag="KILN_ZONE1_TEMP",
            input_tags=["NOPE"],
            lags=[1],
            rolling_windows=[3],
            horizon=1,
        )


def _synthetic_regression(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic linear-ish dataset with three informative features."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f_a": rng.normal(size=n),
            "f_b": rng.normal(size=n),
            "f_c": rng.normal(size=n),
        }
    )
    y = pd.Series(
        3.0 * X["f_a"] - 2.0 * X["f_b"] + 0.5 * X["f_c"] + rng.normal(scale=0.01, size=n),
        name="kpi",
    )
    return X, y


# ---------------------------------------------------------------------------
# EnergyKPIModel
# ---------------------------------------------------------------------------
def test_model_fits_and_preserves_feature_order():
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    assert model.is_fitted
    assert model.feature_cols == ["f_a", "f_b", "f_c"]
    assert model.metadata["n_samples"] == len(X)
    assert model.metadata["n_features"] == 3
    assert model.metadata["feature_names"] == ["f_a", "f_b", "f_c"]
    assert model.metadata["target_name"] == "kpi"
    assert "trained_at" in model.metadata


def test_predict_reorders_columns():
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    shuffled = X[["f_c", "f_a", "f_b"]]
    preds_shuffled = model.predict(shuffled)
    preds_ordered = model.predict(X)
    assert np.allclose(preds_shuffled, preds_ordered)
    assert preds_ordered.shape == (len(X),)


def test_predict_before_fit_raises():
    model = EnergyKPIModel()
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"f_a": [0.0]}))


def test_predict_missing_feature_raises():
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X.drop(columns=["f_b"]))


def test_save_load_roundtrip(tmp_path):
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    path = tmp_path / "nested" / "model.joblib"
    model.save(str(path))
    assert path.exists()

    loaded = EnergyKPIModel.load(str(path))
    assert isinstance(loaded, EnergyKPIModel)
    assert loaded.feature_cols == model.feature_cols
    assert loaded.metadata == model.metadata
    assert np.allclose(loaded.predict(X), model.predict(X))


def test_feature_importance_shape_and_sorting():
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    fi = model.get_feature_importance()
    assert list(fi.columns) == ["feature", "importance"]
    assert len(fi) == 3
    assert set(fi["feature"]) == {"f_a", "f_b", "f_c"}
    assert fi["importance"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Metrics / evaluation / splitting
# ---------------------------------------------------------------------------
def test_regression_metrics_perfect_prediction():
    y = [1.0, 2.0, 4.0]
    metrics = regression_metrics(y, y)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mape"] == 0.0
    assert all(isinstance(v, float) for v in metrics.values())


def test_regression_metrics_ignores_zero_targets_for_mape():
    y_true = [0.0, 100.0]
    y_pred = [5.0, 110.0]
    metrics = regression_metrics(y_true, y_pred)
    # MAPE only considers the non-zero target: |110-100|/100 * 100 = 10%.
    assert metrics["mape"] == pytest.approx(10.0)


def test_regression_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])


def test_train_valid_split_preserves_order():
    X, y = _synthetic_regression(n=100)
    X_tr, X_va, y_tr, y_va = train_valid_split_time_series(X, y, valid_fraction=0.2)
    assert len(X_tr) == 80 and len(X_va) == 20
    # Validation window is strictly after training window (no shuffling).
    assert X_tr.index.max() < X_va.index.min()
    assert list(X_va.index) == list(range(80, 100))


def test_train_valid_split_invalid_fraction_raises():
    X, y = _synthetic_regression(n=10)
    with pytest.raises(ValueError):
        train_valid_split_time_series(X, y, valid_fraction=1.5)


def test_evaluate_regressor_returns_required_keys():
    X, y = _synthetic_regression()
    model = EnergyKPIModel(target_name="kpi").fit(X, y)
    report = evaluate_regressor(model, X, y)
    for key in ("rmse", "mae", "mape", "n_samples", "n_features", "target_name"):
        assert key in report
    assert report["n_samples"] == len(X)
    assert report["n_features"] == 3
    assert report["target_name"] == "kpi"


# ---------------------------------------------------------------------------
# Training pipeline (src.ml.train)
# ---------------------------------------------------------------------------
_TRAIN_TAGS = [
    "KILN_ZONE1_TEMP",
    "KILN_ZONE2_TEMP",
    "KILN_FUEL_FLOW",
    "KILN_PRIMARY_AIR_FLOW",
    "FEED_RATE",
]


def _in_memory_session():
    """Create a session bound to a fresh in-memory SQLite database."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    return get_session_factory(engine)()


def _populate_signals(session, tags: list[str], periods: int = 80) -> None:
    """Insert deterministic long-format signals for ``tags`` into the DB."""
    index = pd.date_range("2026-01-01 00:00", periods=periods, freq="min")
    frames = []
    for offset, tag in enumerate(tags):
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "tag_name": tag,
                    # Distinct, monotonic-but-noisy series per tag.
                    "value": np.linspace(0, 10, periods) + offset
                    + np.sin(np.arange(periods) / 5.0),
                }
            )
        )
    save_signals_to_db(pd.concat(frames, ignore_index=True), session)


def _valid_train_config() -> dict:
    return {
        "target_tag": "KILN_ZONE1_TEMP",
        "input_tags": _TRAIN_TAGS,
        "lags": [1, 5, 10],
        "rolling_windows": [3, 5, 10],
        "horizon": 3,
        "model_output_path": "artifacts/models/kiln_zone1_temp_model.joblib",
    }


def test_load_training_config_success(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_valid_train_config()))
    cfg = load_training_config(str(cfg_path))
    assert cfg["target_tag"] == "KILN_ZONE1_TEMP"
    assert cfg["horizon"] == 3
    assert cfg["input_tags"] == _TRAIN_TAGS


def test_load_training_config_missing_key_raises(tmp_path):
    bad = _valid_train_config()
    del bad["horizon"]
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="horizon"):
        load_training_config(str(cfg_path))


def test_load_signals_from_db_schema():
    session = _in_memory_session()
    _populate_signals(session, _TRAIN_TAGS, periods=20)
    df = load_signals_from_db(session)
    assert list(df.columns) == SIGNAL_COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert set(df["tag_name"]) == set(_TRAIN_TAGS)


def test_load_signals_from_db_filters_by_tag():
    session = _in_memory_session()
    _populate_signals(session, _TRAIN_TAGS, periods=10)
    df = load_signals_from_db(session, tag_names=["FEED_RATE"])
    assert set(df["tag_name"]) == {"FEED_RATE"}


def test_train_energy_model_end_to_end(tmp_path):
    session = _in_memory_session()
    _populate_signals(session, _TRAIN_TAGS, periods=80)
    out_path = tmp_path / "models" / "kiln_zone1_temp_model.joblib"

    summary = train_energy_model(
        session,
        target_tag="KILN_ZONE1_TEMP",
        input_tags=_TRAIN_TAGS,
        lags=[1, 5, 10],
        rolling_windows=[3, 5, 10],
        horizon=3,
        model_output_path=str(out_path),
    )

    required_keys = {
        "target_tag",
        "n_total_samples",
        "n_train_samples",
        "n_valid_samples",
        "n_features",
        "train_metrics",
        "valid_metrics",
        "model_output_path",
        "model_backend",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["target_tag"] == "KILN_ZONE1_TEMP"
    assert summary["n_total_samples"] == (
        summary["n_train_samples"] + summary["n_valid_samples"]
    )
    assert summary["n_features"] > 0
    assert {"rmse", "mae", "mape"}.issubset(summary["train_metrics"].keys())
    # Artifact must exist and be loadable as an EnergyKPIModel.
    assert out_path.exists()
    reloaded = EnergyKPIModel.load(str(out_path))
    assert reloaded.target_name == "KILN_ZONE1_TEMP"


def test_train_energy_model_empty_db_raises(tmp_path):
    session = _in_memory_session()  # no signals inserted
    with pytest.raises(ValueError, match="No signals found"):
        train_energy_model(
            session,
            target_tag="KILN_ZONE1_TEMP",
            input_tags=_TRAIN_TAGS,
            lags=[1],
            rolling_windows=[3],
            horizon=1,
            model_output_path=str(tmp_path / "model.joblib"),
        )


def test_train_energy_model_missing_tag_propagates_value_error(tmp_path):
    session = _in_memory_session()
    # Only the target tag exists; an input tag is absent from the DB.
    _populate_signals(session, ["KILN_ZONE1_TEMP"], periods=40)
    with pytest.raises(ValueError):
        train_energy_model(
            session,
            target_tag="KILN_ZONE1_TEMP",
            input_tags=["KILN_ZONE1_TEMP", "MISSING_TAG"],
            lags=[1, 5],
            rolling_windows=[3],
            horizon=2,
            model_output_path=str(tmp_path / "model.joblib"),
        )
