"""Cement-mill scheduling under an electrical power quota (MILP).

The problem
-----------
Grinding is the largest electricity consumer of a cement plant and the most
flexible one: a silo decouples mills from dispatch, so mills can run when
power is allowed and cheap and stop when it is not. Under Iran's power
restrictions plants get a time-varying quota on the regulated grid and can
buy extra power on the free market at a higher price. Operators usually run
mills on silo level alone, which wastes the quota and buys free-market
power at the wrong hours.

Formulation (per slot ``t`` of ``dt`` hours, mill ``m``)
--------------------------------------------------------
Decision variables::

    x[m,t] in {0,1}   mill m runs in slot t
    s[m,t] >= 0       mill m starts in slot t   (s >= x[t] - x[t-1])
    L[t]              silo level at the end of slot t (t)
    g[t]  >= 0        power drawn under the regulated quota (kW)
    f[t]  >= 0        power bought on the free market (kW)
    u[t]  >= 0        dispatch that could not be served (t) - penalized

Constraints::

    sum_m kw_m x[m,t] = g[t] + f[t]
    g[t] <= quota[t],  f[t] <= free_cap[t]
    L[t] = L[t-1] + dt * (sum_m tph_m x[m,t] - dispatch[t]) + u[t]
    L_min <= L[t] <= L_max,  L[T-1] >= L_end
    x[m,t] = 0 where the mill is unavailable (maintenance)

Objective::

    min  sum_t dt * (grid_price[t] g[t] + free_price[t] f[t])
       + start_cost * sum s + shortfall_penalty * sum u

Solved with HiGHS through :func:`scipy.optimize.milp`; no extra dependency.
:func:`naive_schedule` reproduces the usual silo-level rule so the saving
can be reported against what the plant does today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Mill:
    """A cement mill as the scheduler sees it."""

    name: str
    rated_kw: float
    rated_tph: float
    start_cost: float = 0.0  # money per start (wear, ramp-up energy, quality)

    def __post_init__(self) -> None:
        if self.rated_kw <= 0 or self.rated_tph <= 0:
            raise ValueError(f"mill {self.name}: rated_kw and rated_tph must be positive.")

    @property
    def kwh_per_t(self) -> float:
        return self.rated_kw / self.rated_tph


@dataclass
class ScheduleProblem:
    """Inputs of one scheduling run. Per-slot arrays share one length."""

    mills: List[Mill]
    slot_hours: float
    quota_kw: Sequence[float]
    grid_price: Sequence[float]  # money per kWh
    free_market_price: Sequence[float]  # money per kWh
    free_market_cap_kw: Sequence[float]
    dispatch_tph: Sequence[float]
    silo_initial_t: float
    silo_min_t: float
    silo_max_t: float
    silo_end_min_t: Optional[float] = None  # default: back to the initial level
    availability: Optional[np.ndarray] = None  # bool (n_mills, n_slots)
    shortfall_penalty: float = 1e6  # money per tonne not dispatched
    start_index: Optional[pd.DatetimeIndex] = None

    def __post_init__(self) -> None:
        n = len(self.quota_kw)
        for name in ("grid_price", "free_market_price", "free_market_cap_kw", "dispatch_tph"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have {n} slots like quota_kw.")
        if not self.mills:
            raise ValueError("at least one mill is required.")
        if self.slot_hours <= 0:
            raise ValueError("slot_hours must be positive.")
        if not self.silo_min_t <= self.silo_initial_t <= self.silo_max_t:
            raise ValueError("silo_initial_t must lie within [silo_min_t, silo_max_t].")
        if self.availability is not None and self.availability.shape != (len(self.mills), n):
            raise ValueError("availability must have shape (n_mills, n_slots).")

    @property
    def n_slots(self) -> int:
        return len(self.quota_kw)


@dataclass
class Schedule:
    """A solved (or simulated) schedule with its cost breakdown."""

    running: np.ndarray  # bool (n_mills, n_slots)
    silo_t: np.ndarray
    grid_kw: np.ndarray
    free_kw: np.ndarray
    shortfall_t: np.ndarray
    energy_cost: float
    start_cost: float
    n_starts: int
    status: str
    mills: List[Mill] = field(repr=False, default_factory=list)
    slot_hours: float = 1.0
    index: Optional[pd.DatetimeIndex] = None

    @property
    def energy_kwh(self) -> float:
        return float((self.grid_kw + self.free_kw).sum() * self.slot_hours)

    @property
    def free_market_kwh(self) -> float:
        return float(self.free_kw.sum() * self.slot_hours)

    @property
    def cement_t(self) -> float:
        tph = np.array([m.rated_tph for m in self.mills])[:, None]
        return float((self.running * tph).sum() * self.slot_hours)

    @property
    def kwh_per_t(self) -> float:
        return self.energy_kwh / self.cement_t if self.cement_t > 0 else float("nan")

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.start_cost

    def to_frame(self) -> pd.DataFrame:
        """One row per slot, one column per mill (1 = running)."""
        data = {m.name: self.running[i].astype(int) for i, m in enumerate(self.mills)}
        data.update(
            {
                "silo_t": self.silo_t,
                "grid_kw": self.grid_kw,
                "free_market_kw": self.free_kw,
                "shortfall_t": self.shortfall_t,
            }
        )
        return pd.DataFrame(data, index=self.index)

    def summary(self) -> dict:
        return {
            "status": self.status,
            "energy_kwh": self.energy_kwh,
            "free_market_kwh": self.free_market_kwh,
            "cement_t": self.cement_t,
            "kwh_per_t": self.kwh_per_t,
            "energy_cost": self.energy_cost,
            "start_cost": self.start_cost,
            "total_cost": self.total_cost,
            "n_starts": self.n_starts,
            "shortfall_t": float(self.shortfall_t.sum()),
            "min_silo_t": float(self.silo_t.min()),
        }


def _costs(problem: ScheduleProblem, grid_kw: np.ndarray, free_kw: np.ndarray) -> float:
    dt = problem.slot_hours
    return float(
        dt
        * (
            np.dot(np.asarray(problem.grid_price, float), grid_kw)
            + np.dot(np.asarray(problem.free_market_price, float), free_kw)
        )
    )


def _count_starts(running: np.ndarray, initially_running: Sequence[bool]) -> np.ndarray:
    prev = np.column_stack([np.asarray(initially_running, bool), running[:, :-1]])
    return (running & ~prev).astype(int)


def optimize_schedule(
    problem: ScheduleProblem,
    initially_running: Optional[Sequence[bool]] = None,
    time_limit_s: float = 30.0,
    mip_rel_gap: float = 1e-4,
) -> Schedule:
    """Solve the scheduling MILP.

    Raises
    ------
    RuntimeError
        If the solver finds no feasible schedule (it always should, thanks to
        the shortfall slack, unless the silo bounds are contradictory).
    """
    M, N = len(problem.mills), problem.n_slots
    dt = problem.slot_hours
    init = np.asarray(initially_running if initially_running is not None else [False] * M, bool)
    end_min = problem.silo_initial_t if problem.silo_end_min_t is None else problem.silo_end_min_t

    # Variable layout.
    ix = lambda m, t: m * N + t  # noqa: E731
    i_s = M * N
    i_L = 2 * M * N
    i_g = i_L + N
    i_f = i_g + N
    i_u = i_f + N
    n_var = i_u + N

    c = np.zeros(n_var)
    c[i_s : i_s + M * N] = np.repeat([m.start_cost for m in problem.mills], N)
    c[i_g : i_g + N] = dt * np.asarray(problem.grid_price, float)
    c[i_f : i_f + N] = dt * np.asarray(problem.free_market_price, float)
    c[i_u : i_u + N] = problem.shortfall_penalty

    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    ub[: M * N] = 1.0
    if problem.availability is not None:
        ub[: M * N] = problem.availability.astype(float).ravel()
    lb[i_L : i_L + N] = problem.silo_min_t
    ub[i_L : i_L + N] = problem.silo_max_t
    lb[i_L + N - 1] = max(problem.silo_min_t, min(end_min, problem.silo_max_t))
    ub[i_g : i_g + N] = np.asarray(problem.quota_kw, float)
    ub[i_f : i_f + N] = np.asarray(problem.free_market_cap_kw, float)

    integrality = np.zeros(n_var)
    integrality[: M * N] = 1

    rows = 2 * N + M * N
    A = lil_matrix((rows, n_var))
    lo = np.zeros(rows)
    hi = np.zeros(rows)
    r = 0
    dispatch = np.asarray(problem.dispatch_tph, float)
    for t in range(N):
        # Power balance: sum kw x - g - f = 0
        for m, mill in enumerate(problem.mills):
            A[r, ix(m, t)] = mill.rated_kw
        A[r, i_g + t] = -1.0
        A[r, i_f + t] = -1.0
        r += 1
        # Silo balance: L[t] - L[t-1] - dt*sum tph x - u[t] = -dt*dispatch
        A[r, i_L + t] = 1.0
        if t > 0:
            A[r, i_L + t - 1] = -1.0
        for m, mill in enumerate(problem.mills):
            A[r, ix(m, t)] = -dt * mill.rated_tph
        A[r, i_u + t] = -1.0
        rhs = -dt * dispatch[t] + (problem.silo_initial_t if t == 0 else 0.0)
        lo[r] = hi[r] = rhs
        r += 1
    for m in range(M):
        for t in range(N):
            # Start indicator: s[m,t] - x[m,t] + x[m,t-1] >= 0 (x[-1] = init)
            A[r, i_s + ix(m, t)] = 1.0
            A[r, ix(m, t)] = -1.0
            if t > 0:
                A[r, ix(m, t - 1)] = 1.0
                lo[r] = 0.0
            else:
                lo[r] = -1.0 * float(init[m])
            hi[r] = np.inf
            r += 1

    result = milp(
        c,
        constraints=LinearConstraint(A.tocsr(), lo, hi),
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": time_limit_s, "mip_rel_gap": mip_rel_gap, "disp": False},
    )
    if result.x is None:
        raise RuntimeError(f"Mill scheduling failed: {result.message}")
    x = result.x
    running = np.round(x[: M * N]).reshape(M, N).astype(bool)
    grid = np.clip(x[i_g : i_g + N], 0, None)
    free = np.clip(x[i_f : i_f + N], 0, None)
    starts = _count_starts(running, init)
    start_cost = float(sum(starts[m].sum() * mill.start_cost for m, mill in enumerate(problem.mills)))
    return Schedule(
        running=running,
        silo_t=x[i_L : i_L + N],
        grid_kw=grid,
        free_kw=free,
        shortfall_t=np.clip(x[i_u : i_u + N], 0, None),
        energy_cost=_costs(problem, grid, free),
        start_cost=start_cost,
        n_starts=int(starts.sum()),
        status="optimal" if result.status == 0 else str(result.message),
        mills=list(problem.mills),
        slot_hours=dt,
        index=problem.start_index,
    )


def naive_schedule(
    problem: ScheduleProblem,
    start_below: Sequence[float],
    stop_above: Sequence[float],
    initially_running: Optional[Sequence[bool]] = None,
    buy_free_market: bool = True,
) -> Schedule:
    """Silo-level hysteresis rule: what most plants do today.

    Each mill starts when silo fill falls below ``start_below[m]`` and stops
    above ``stop_above[m]``. Power beyond the quota is bought on the free
    market (if allowed and within its cap); otherwise mills are shed in order.
    """
    M, N = len(problem.mills), problem.n_slots
    dt = problem.slot_hours
    initial = list(initially_running) if initially_running is not None else [False] * M
    wants = list(initial)
    running = np.zeros((M, N), bool)
    silo = np.zeros(N)
    grid = np.zeros(N)
    free = np.zeros(N)
    short = np.zeros(N)
    level = problem.silo_initial_t
    span = problem.silo_max_t
    for t in range(N):
        fill = level / span
        for m in range(M):
            if fill < start_below[m]:
                wants[m] = True
            elif fill > stop_above[m]:
                wants[m] = False
        quota = float(problem.quota_kw[t])
        cap = float(problem.free_market_cap_kw[t]) if buy_free_market else 0.0
        used = 0.0
        for m, mill in enumerate(problem.mills):
            if problem.availability is not None and not problem.availability[m, t]:
                continue
            if wants[m] and used + mill.rated_kw <= quota + cap:
                running[m, t] = True
                used += mill.rated_kw
        grid[t] = min(used, quota)
        free[t] = max(0.0, used - quota)
        produced = sum(problem.mills[m].rated_tph for m in range(M) if running[m, t]) * dt
        level = level + produced - dt * float(problem.dispatch_tph[t])
        if level < problem.silo_min_t:
            short[t] = problem.silo_min_t - level
            level = problem.silo_min_t
        level = min(level, problem.silo_max_t)
        silo[t] = level
    starts = _count_starts(running, initial)
    start_cost = float(sum(starts[m].sum() * mill.start_cost for m, mill in enumerate(problem.mills)))
    return Schedule(
        running=running,
        silo_t=silo,
        grid_kw=grid,
        free_kw=free,
        shortfall_t=short,
        energy_cost=_costs(problem, grid, free),
        start_cost=start_cost,
        n_starts=int(starts.sum()),
        status="simulated",
        mills=list(problem.mills),
        slot_hours=dt,
        index=problem.start_index,
    )


def compare(optimized: Schedule, baseline: Schedule) -> dict:
    """Savings of the optimized schedule against the baseline rule."""
    return {
        "cost_saving": baseline.total_cost - optimized.total_cost,
        "cost_saving_pct": (
            100.0 * (baseline.total_cost - optimized.total_cost) / baseline.total_cost
            if baseline.total_cost > 0
            else 0.0
        ),
        "free_market_kwh_avoided": baseline.free_market_kwh - optimized.free_market_kwh,
        "starts_avoided": baseline.n_starts - optimized.n_starts,
        "shortfall_t_baseline": float(baseline.shortfall_t.sum()),
        "shortfall_t_optimized": float(optimized.shortfall_t.sum()),
        "optimized": optimized.summary(),
        "baseline": baseline.summary(),
    }


def like_for_like(
    problem: ScheduleProblem,
    start_below: Sequence[float],
    stop_above: Sequence[float],
    initially_running: Optional[Sequence[bool]] = None,
    time_limit_s: float = 30.0,
) -> dict:
    """Run today's rule, then optimize to end with at least the same silo level.

    Comparing against a rule that ends with a fuller silo would credit the
    optimizer for simply grinding less; this forces equal or more cement in
    stock at the end of the horizon, so the saving is real.
    """
    baseline = naive_schedule(problem, start_below, stop_above, initially_running)
    from dataclasses import replace

    fair = replace(problem, silo_end_min_t=float(baseline.silo_t[-1]))
    optimized = optimize_schedule(fair, initially_running, time_limit_s=time_limit_s)
    result = compare(optimized, baseline)
    result["silo_end_t"] = {
        "baseline": float(baseline.silo_t[-1]),
        "optimized": float(optimized.silo_t[-1]),
    }
    result["cost_per_t"] = {
        "baseline": baseline.total_cost / baseline.cement_t if baseline.cement_t else None,
        "optimized": optimized.total_cost / optimized.cement_t if optimized.cement_t else None,
    }
    return {"comparison": result, "optimized": optimized, "baseline": baseline}
