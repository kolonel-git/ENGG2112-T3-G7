import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.data.clean import (
    add_outage_flag,
    add_suspect_wind_flag,
    clean_year,
    dedupe_keys,
    interval_start,
    join_tables,
    mask_sentinels,
)
from wt_predmaint.data.daily import add_daily_flags
from wt_predmaint.data.scada import key_columns

# All frames below are synthetic.


def frame(values: dict[str, list[float]], stamps: list[str], station: int = 1) -> pd.DataFrame:
    df = pd.DataFrame(values)
    df.insert(0, "TimeStamp", pd.to_datetime(stamps))
    df.insert(0, "StationId", station)
    return df


def test_interval_start_is_ten_minutes_earlier() -> None:
    s = pd.Series(pd.to_datetime(["2022-01-01 00:00:00"]))
    assert interval_start(s).iloc[0] == pd.Timestamp("2021-12-31 23:50:00")


def test_dedupe_counts_exact_and_conflicting() -> None:
    df = frame({"a": [1.0, 1.0, 2.0, 5.0]}, ["2020-01-01 00:10"] * 2 + ["2020-01-01 00:20"] * 2)
    out, exact, conflict = dedupe_keys(df)
    assert (len(out), exact, conflict) == (2, 1, 1)
    assert out["a"].tolist() == [1.0, 2.0]


def test_mask_sentinels_counts_per_row_and_keeps_other_values() -> None:
    df = frame({"t": [20.0, 200.0, -50.0], "u": [200.0, 21.0, 22.0]}, ["2020-01-01 00:10"] * 3)
    out, n = mask_sentinels(df, [200.0, -50.0])
    assert out["t"].isna().tolist() == [False, True, True]
    assert out["u"].isna().tolist() == [True, False, False]
    assert n.tolist() == [1, 1, 1]
    assert df["t"].tolist() == [20.0, 200.0, -50.0]  # input untouched


def test_mask_nothing_when_no_sentinels() -> None:
    df = frame({"p": [200.0]}, ["2020-01-01 00:10"])
    out, n = mask_sentinels(df, [])
    assert out["p"].iloc[0] == 200.0 and n.iloc[0] == 0


def test_join_inner_dedupes_and_sums_masked() -> None:
    stamps = ["2020-01-01 00:10", "2020-01-01 00:20"]
    temp = frame({"t": [200.0, 30.0]}, stamps)
    press = frame({"p": [5.0, -100.0]}, stamps + [])
    press = pd.concat([press, press.iloc[[1]]], ignore_index=True)  # exact duplicate row
    turb = frame({"w": [7.0]}, stamps[:1])  # second timestamp missing -> inner join drops it
    out, stats = join_tables(
        {"tblSCTurbine": turb, "tblSCTurTemp": temp, "tblSCTurPress": press},
        {"tblSCTurTemp": [200.0], "tblSCTurPress": [-100.0]},
    )
    assert len(out) == 1
    assert out["n_masked"].iloc[0] == 1  # only the temperature sentinel on the kept row
    assert stats["tables"]["tblSCTurPress"]["exact_duplicates"] == 1
    assert stats["rows_joined"] == 1


def test_outage_month_flag() -> None:
    full = pd.date_range("2020-01-01 00:10", "2020-02-01 00:00", freq="10min")  # all of January
    part = pd.date_range("2020-02-01 00:10", periods=300, freq="10min")  # sparse February
    df = pd.DataFrame({"StationId": 1, "TimeStamp": full.append(part)})
    out, bad = add_outage_flag(df, expected_turbines=1, coverage_threshold=0.9)
    assert bad == ["2020-02"]
    assert not out.loc[out.TimeStamp <= "2020-02-01 00:00", "outage_month"].any()
    assert out.loc[out.TimeStamp > "2020-02-01 00:00", "outage_month"].all()


def test_suspect_wind_flag_only_for_uncorrelated_turbine() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    rpm = rng.uniform(1100, 1500, n)
    good = pd.DataFrame(
        {
            "StationId": 1,
            "wtc_GenRpm_mean": rpm,
            "wtc_PrWindSp_mean": rpm / 150 + rng.normal(0, 0.1, n),
        }
    )
    bad = pd.DataFrame(
        {"StationId": 2, "wtc_GenRpm_mean": rpm, "wtc_PrWindSp_mean": rng.uniform(3, 12, n)}
    )
    out, suspect = add_suspect_wind_flag(pd.concat([good, bad]), 1000, 0.55, 1000)
    assert [s for s, _ in suspect] == [2]
    assert out.groupby("StationId")["suspect_wind"].all().to_dict() == {1: False, 2: True}


def test_clean_year_assigns_boundary_row_to_previous_year() -> None:
    cfg = load_protocol()
    cfg["cleaning"]["expected_turbines"] = 1
    stamps = ["2021-12-31 23:50:00", "2022-01-01 00:00:00", "2022-01-01 00:10:00"]
    tables = {
        "tblSCTurbine": frame(
            {"wtc_GenRpm_mean": [1200.0] * 3, "wtc_PrWindSp_mean": [8.0] * 3}, stamps
        ),
        "tblSCTurTemp": frame({"t": [30.0] * 3}, stamps),
        "tblSCTurPress": frame({"p": [220.0] * 3}, stamps),
    }
    y2021, _ = clean_year(tables, 2021, cfg)
    y2022, _ = clean_year(tables, 2022, cfg)
    assert (
        y2021["TimeStamp"].tolist() == pd.to_datetime(stamps[:2]).tolist()
    )  # 00:00 ends a 2021 interval
    assert y2022["TimeStamp"].tolist() == pd.to_datetime(stamps[2:]).tolist()


def test_daily_flags() -> None:
    cfg = load_protocol()
    df = pd.DataFrame(
        {
            "Energy": [100.0, -5.0, 50.0],
            "Samples": [100, 100, 40],
            "SamplePeriode": [1440, 1440, 1440],
            "TimeOk": [24.0, 24.0, 30.0],
            "TimeReady": [24.0, 24.0, 24.0],
            "TimeErr": [0.0, 0.0, 0.0],
            "TimeExternalErr": [0.0, 0.0, 0.0],
            "TimeGenerator": [10.0, 10.0, 10.0],
        }
    )
    out = add_daily_flags(df, cfg)
    assert out["negative_energy"].tolist() == [False, True, False]
    assert out["partial_day"].tolist() == [False, False, True]
    assert out["bad_hours"].tolist() == [False, False, True]


def test_key_columns_keep_min_max_only_for_key_sensors() -> None:
    header = [
        "TimeStamp",
        "StationId",
        "wtc_GeOilTmp_min",
        "wtc_GeOilTmp_max",
        "wtc_GeOilTmp_mean",
        "wtc_NMMBrdTm_min",
        "wtc_NMMBrdTm_mean",
        "wtc_GenRpm_stddev",
        "wtc_PrWindSp_stddev",
        "wtc_HSGenTmp_stddev",
    ]
    cols = key_columns(header)
    assert "wtc_GeOilTmp_min" in cols and "wtc_GeOilTmp_max" in cols
    assert "wtc_NMMBrdTm_min" not in cols and "wtc_NMMBrdTm_mean" in cols
    assert "wtc_PrWindSp_stddev" in cols and "wtc_GenRpm_stddev" in cols
    assert "wtc_HSGenTmp_stddev" not in cols  # stddev is kept only for wind speed and rpm


def test_clean_year_requires_cleaning_config() -> None:
    with pytest.raises(KeyError):
        clean_year({}, 2020, {})
