import pandas as pd
import pytest

from wt_predmaint.config import load_protocol, require
from wt_predmaint.data.alarms import (
    alarm_code_table,
    build_alarm_events,
    count_summary,
    group_episodes,
    load_alarm_log,
    other_alarm_events,
    read_alarm_log,
)

# Synthetic rows in the documented tblAlarmLog layout; not real data.
CSV = (
    "TimeOn,TimeOff,StationNr,Alarmcode\n"
    "2018-03-01 10:07:30,2018-03-01 10:12:00,2304510,3130\n"
    "2018-03-01 10:07:30,2018-03-01 10:12:00,2304510,3130\n"  # exact duplicate
    "2018-03-02 11:00:00,,2304510,20\n"
    "2018-03-03 12:00:00,,2304511,127\n"  # out of scope
    "2018-03-03 12:00:00,,91,50358\n"  # site-level station
)


@pytest.fixture
def codes() -> pd.DataFrame:
    return alarm_code_table(load_protocol())


@pytest.fixture
def raw() -> pd.DataFrame:
    return load_alarm_log([("a.csv", CSV.encode())])


def test_code_table_has_twelve_codes_and_tiers(codes: pd.DataFrame) -> None:
    assert len(codes) == 12
    assert set(codes.loc[codes.Tier == 1, "Alarmcode"]) == {3130, 10105}
    assert set(codes.loc[codes.Tier == 3, "Alarmcode"]) == {20, 25, 102}


def test_load_drops_site_station_and_duplicates(raw: pd.DataFrame) -> None:
    assert set(raw.StationId) == {2304510, 2304511}
    assert len(raw) == 3


def test_build_events_filters_and_floors(raw: pd.DataFrame, codes: pd.DataFrame) -> None:
    ev = build_alarm_events(raw, codes)
    assert set(ev.Alarmcode) == {3130, 20}
    first = ev[ev.Alarmcode == 3130].iloc[0]
    assert first.TimeStamp == pd.Timestamp("2018-03-01 10:10:00")  # interval-end label
    assert first.DurationMin == pytest.approx(4.5)
    assert first.Tier == 1 and bool(first.Stopping)
    assert pd.isna(ev[ev.Alarmcode == 20].iloc[0].DurationMin)


def test_other_events_kept_separately(raw: pd.DataFrame, codes: pd.DataFrame) -> None:
    other = other_alarm_events(raw, codes)
    assert list(other.Alarmcode) == [127]


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        read_alarm_log("bad.csv", b"a,b\n1,2\n")


def test_count_summary(raw: pd.DataFrame, codes: pd.DataFrame) -> None:
    summary = count_summary(build_alarm_events(raw, codes))
    assert summary.loc[(1, 3130), 2018] == 1
    assert summary.loc[(3, 20), "Turbines"] == 1


def test_group_episodes_merges_close_events() -> None:
    t = pd.Timestamp
    ev = pd.DataFrame(
        {
            "StationId": [1, 1, 1, 2],
            "Alarmcode": [3130, 3130, 3130, 3130],
            "Tier": [1] * 4,
            "Stopping": [True] * 4,
            "TimeOn": [t("2018-03-01"), t("2018-03-03"), t("2018-04-20"), t("2018-03-01")],
            "TimeOff": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    ep = group_episodes(ev, pd.Timedelta(days=7))
    assert list(ep.loc[ep.StationId == 1, "Events"]) == [2, 1]
    assert len(ep.loc[ep.StationId == 2]) == 1


def test_require_rejects_todo() -> None:
    with pytest.raises(ValueError, match="TODO"):
        require({"labels": {"x": "TODO"}}, "labels.x")
    with pytest.raises(KeyError):
        require({"labels": {}}, "labels.missing")
    assert require(load_protocol(), "labels.positive_window_days") == 30
    assert require(load_protocol(), "labels.feature_tiers") == [1, 2, 3]


def test_month_first_slash_timestamps_parse() -> None:
    csv = (
        "TimeOn,TimeOff,StationNr,Alarmcode\n"
        "1/13/2025 0:21,1/13/2025 3:45,2304510,3130\n"
        "1/2/2025 10:05,,2304510,20\n"
    )
    df = read_alarm_log("slash.csv", csv.encode())
    assert df.TimeOn.iloc[0] == pd.Timestamp("2025-01-13 00:21")
    assert df.TimeOn.iloc[1] == pd.Timestamp("2025-01-02 10:05")
    assert df.TimeOff.iloc[0] == pd.Timestamp("2025-01-13 03:45")


def test_unknown_timestamp_format_raises() -> None:
    bad = b"TimeOn,TimeOff,StationNr,Alarmcode\n13-01-2025,,1,20\n"
    with pytest.raises(ValueError, match="unrecognised timestamp"):
        read_alarm_log("bad.csv", bad)


def test_iso_dates_with_and_without_time_parse_together() -> None:
    from wt_predmaint.data.alarms import parse_times

    out = parse_times(pd.Series(["2022-10-31 23:50:00", "2022-11-01"]), "mixed.csv")
    assert out.iloc[1] == pd.Timestamp("2022-11-01 00:00:00")


def test_month_first_date_only_parses() -> None:
    from wt_predmaint.data.alarms import parse_times

    out = parse_times(pd.Series(["1/13/2026", "2/1/2026"]), "daily.csv")
    assert out.tolist() == [pd.Timestamp("2026-01-13"), pd.Timestamp("2026-02-01")]
