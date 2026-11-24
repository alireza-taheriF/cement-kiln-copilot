"""Transport-delay (dead-time) estimation between process tags.

A kiln reacts to a fuel change only after tens of minutes. A model that only
sees the last few rows of fuel flow cannot learn that, and a recommendation
engine built on it will credit the wrong input. This module estimates the
delay from each input to the target by cross-correlating the *changes* of the two
series over ``diff_rows`` rows. Differencing removes slow trends that would
otherwise dominate the correlation; differencing over several rows rather
than one keeps the low-pass response of a kiln (whose one-step changes are
buried in sensor noise) visible.

The estimate is a starting point for feature design, not a system
identification. Delays are reported in rows of the regular grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DelayEstimate:
    """Delay from one input to the target."""

    tag: str
    delay_rows: int
    correlation: float  # signed correlation of differenced series at that delay
    n_samples: int

    @property
    def is_reliable(self) -> bool:
        """Heuristic: a weak or under-sampled peak should not drive features."""
        return abs(self.correlation) >= 0.1 and self.n_samples >= 50


def estimate_delay(
    x: pd.Series,
    y: pd.Series,
    max_delay: int,
    min_delay: int = 0,
    diff_rows: int = 3,
) -> DelayEstimate:
    """Delay ``d`` (rows) maximizing ``|corr(Δx[t - d], Δy[t])|``.

    ``Δ`` is the change over ``diff_rows`` rows.

    Parameters
    ----------
    x, y:
        Input and target on the same regular index.
    max_delay:
        Largest delay considered (rows).
    min_delay:
        Smallest delay considered (rows), default 0.
    diff_rows:
        Differencing span in rows (>= 1).
    """
    if max_delay < min_delay:
        raise ValueError("max_delay must be >= min_delay.")
    if diff_rows < 1:
        raise ValueError("diff_rows must be >= 1.")
    frame = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    dx = frame["x"].diff(diff_rows)
    dy = frame["y"].diff(diff_rows)
    best_d, best_c, best_n = min_delay, 0.0, 0
    for d in range(min_delay, max_delay + 1):
        pair = pd.concat([dx.shift(d), dy], axis=1).dropna()
        if len(pair) < 3:
            continue
        a = pair.iloc[:, 0].to_numpy()
        b = pair.iloc[:, 1].to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if abs(c) > abs(best_c):
            best_d, best_c, best_n = d, c, len(pair)
    return DelayEstimate(str(x.name), best_d, best_c, best_n)


def estimate_delays(
    wide: pd.DataFrame,
    target_tag: str,
    input_tags: Iterable[str],
    max_delay: int,
    diff_rows: int = 3,
) -> Dict[str, DelayEstimate]:
    """Delay estimate for every input tag (the target's own delay is 0)."""
    if target_tag not in wide:
        raise ValueError(f"target tag {target_tag!r} not in frame.")
    out: Dict[str, DelayEstimate] = {}
    for tag in input_tags:
        if tag not in wide:
            raise ValueError(f"input tag {tag!r} not in frame.")
        if tag == target_tag:
            out[tag] = DelayEstimate(tag, 0, 1.0, int(wide[tag].notna().sum()))
            continue
        out[tag] = estimate_delay(
            wide[tag].rename(tag), wide[target_tag], max_delay, diff_rows=diff_rows
        )
        logger.info(
            "Delay %s -> %s: %d rows (corr %.2f)",
            tag,
            target_tag,
            out[tag].delay_rows,
            out[tag].correlation,
        )
    return out


def lags_from_delays(
    delays: Dict[str, DelayEstimate],
    horizon: int,
    base_lags: List[int],
) -> Dict[str, List[int]]:
    """Per-tag lag sets that include the lag matching each tag's dead time.

    With ``y[t] = target[t + horizon]``, an input that acts after ``d`` rows
    is seen at feature lag ``d - horizon`` (clipped at 0, i.e. the current
    value). The ``base_lags`` keep short-term dynamics for every tag.
    """
    out: Dict[str, List[int]] = {}
    for tag, est in delays.items():
        lags = set(int(lag) for lag in base_lags)
        if est.is_reliable:
            lags.add(max(0, est.delay_rows - horizon))
        out[tag] = sorted(lags)
    return out
