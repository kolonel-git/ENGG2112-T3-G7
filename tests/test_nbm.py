"""NBM behaviour on synthetic data with a known linear law; no real data is read."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.nbm import NormalBehaviourModel, indicator, short_name, smooth_residuals

WIND, RPM, SRPM, AMB, NAC = (
    "op_wtc_PrWindSp_mean",
    "op_wtc_GenRpm_mean",
    "op_wtc_MainSRpm_mean",
    "op_wtc_AmbieTmp_mean",
    "op_wtc_NacelTmp_mean",
)
TARGET = "op_wtc_GeOilTmp_mean"


def make_cfg() -> dict:
    cfg = copy.deepcopy(load_protocol())
    cfg["nbm"].update(
        targets=[TARGET], ridge_alpha=1e-6, minimum_fit_rows=20, minimum_turbine_rows=20
    )
    return cfg


def make_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for station, offset in ((1, 10.0), (2, 25.0)):
        days = pd.date_range("2018-01-01", periods=n)
        x = {c: rng.normal(size=n) for c in (WIND, RPM, SRPM, AMB, NAC)}
        y = 2.0 * x[WIND] - 1.0 * x[AMB] + 0.5 * x[NAC] + offset + rng.normal(scale=0.01, size=n)
        parts.append(pd.DataFrame({"StationId": station, "Day": days, **x, TARGET: y}))
    return pd.concat(parts, ignore_index=True)


def fitted() -> tuple[NormalBehaviourModel, pd.DataFrame]:
    frame = make_frame()
    model = NormalBehaviourModel(make_cfg()).fit(frame, pd.Series(True, index=frame.index))
    return model, frame


def test_recovers_linear_law_and_turbine_offsets() -> None:
    model, frame = fitted()
    pred = model.predict(frame)[TARGET]
    assert np.abs(pred - frame[TARGET]).max() < 0.1
    params = model.parameters().iloc[0]
    assert params["coef_PrWindSp"] == pytest.approx(2.0, abs=0.05)
    assert params["coef_AmbieTmp"] == pytest.approx(-1.0, abs=0.05)
    assert params["r2_train"] > 0.99


def test_residual_sign_hotter_than_expected_is_negative() -> None:
    model, frame = fitted()
    hot = frame.copy()
    hot[TARGET] += 5.0
    assert (model.residuals(hot)[f"nbm_res_{short_name(TARGET)}"] < -10).all()


def test_residual_does_not_depend_on_other_rows() -> None:
    model, frame = fitted()
    full = model.residuals(frame)
    half = model.residuals(frame.iloc[:30])
    assert np.allclose(full.iloc[:30], half, equal_nan=True)


def test_unhealthy_rows_are_not_used_for_fitting() -> None:
    frame = make_frame()
    broken = frame.copy()
    broken.loc[broken.index[:40], TARGET] += 100.0
    mask = pd.Series(True, index=frame.index)
    mask.iloc[:40] = False
    clean = NormalBehaviourModel(make_cfg()).fit(frame, mask).predict(frame)
    perturbed = NormalBehaviourModel(make_cfg()).fit(broken, mask).predict(frame)
    assert np.allclose(clean, perturbed)


def test_rejects_rows_outside_train_years() -> None:
    frame = make_frame()
    frame["Day"] = frame["Day"] + pd.DateOffset(years=4)  # 2022: validation
    with pytest.raises(ValueError, match="train years"):
        NormalBehaviourModel(make_cfg()).fit(frame, pd.Series(True, index=frame.index))


def test_missing_input_gives_nan_not_a_score() -> None:
    model, frame = fitted()
    gap = frame.copy()
    gap.loc[0, AMB] = np.nan
    res = model.residuals(gap)[f"nbm_res_{short_name(TARGET)}"]
    assert np.isnan(res.iloc[0]) and res.iloc[1:].notna().all()


def test_unseen_turbine_uses_fleet_offset_and_spread() -> None:
    model, frame = fitted()
    new = frame.iloc[:5].assign(StationId=99)
    assert np.isfinite(model.residuals(new).to_numpy()).all()


def test_smoothing_is_trailing_only() -> None:
    model, frame = fitted()
    res = model.residuals(frame)
    keys = frame[["StationId", "Day"]]
    base = smooth_residuals(res, keys, window=7, min_days=4)
    future = res.copy()
    cutoff = frame["Day"].iloc[0] + pd.Timedelta(days=60)
    future.loc[frame["Day"] > cutoff, :] = 1000.0
    changed = smooth_residuals(future, keys, window=7, min_days=4)
    early = frame["Day"] <= cutoff
    assert np.allclose(base[early], changed[early], equal_nan=True)


def test_smoothing_needs_min_days_and_treats_gaps_as_missing() -> None:
    keys = pd.DataFrame({"StationId": 1, "Day": pd.to_datetime(["2018-01-01", "2018-01-10"])})
    res = pd.DataFrame({"nbm_res_A": [1.0, 3.0]})
    out = smooth_residuals(res, keys, window=7, min_days=2)
    assert out["nbm_w7_A"].isna().all()


def test_indicator_is_max_absolute_value() -> None:
    smoothed = pd.DataFrame({"nbm_w7_A": [1.0, -3.0], "nbm_w7_B": [-2.0, 1.0]})
    assert indicator(smoothed).tolist() == [2.0, 3.0]


def test_fleet_relative_removes_common_drift_and_uses_same_day_only() -> None:
    from wt_predmaint.models.nbm import fleet_relative

    days = pd.date_range("2018-01-01", periods=3)
    keys = pd.DataFrame({"StationId": [1, 2, 3] * 3, "Day": np.repeat(days, 3)})
    res = pd.DataFrame({"nbm_res_A": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0, 5.0, 5.0, 5.0]})
    out = fleet_relative(res, keys, min_turbines=3)["nbm_res_A"]
    assert out.tolist() == [-1.0, 0.0, 1.0] * 2 + [0.0, 0.0, 0.0]
    # a later day cannot change an earlier day's value
    later = res.copy()
    later.iloc[6:] = 1000.0
    assert np.allclose(fleet_relative(later, keys, 3)["nbm_res_A"].iloc[:6], out.iloc[:6])
    # too few turbines on a day gives missing, not a one-turbine "median"
    assert fleet_relative(res.iloc[:2], keys.iloc[:2], 3)["nbm_res_A"].isna().all()


def test_nbm_lookahead_check_passes_and_catches_a_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    import wt_predmaint.models.nbm as nbm_module

    cfg = make_cfg()
    cfg["nbm"].update(fleet_relative=True, fleet_min_turbines=2)
    model, frame = fitted()
    cases = [(1, pd.Timestamp("2018-03-01")), (2, pd.Timestamp("2018-04-10"))]
    assert nbm_module.check_nbm_no_lookahead(model, frame, cfg, cases) == []

    real = nbm_module.nbm_features

    def leaky(m, f, c, fleet=None):  # adds a column built from the whole table (the future)
        out = real(m, f, c, fleet)
        out["nbm_leak"] = float(f[TARGET].mean())
        return out

    monkeypatch.setattr(nbm_module, "nbm_features", leaky)
    problems = nbm_module.check_nbm_no_lookahead(model, frame, cfg, cases)
    assert problems and "nbm_leak" in problems[0]["columns"]
