import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from wt_predmaint.evaluation.plotting import (  # noqa: E402
    NO_DATA,
    blank_reasons,
    calendar_frame,
    plot_risk_timeline,
    status_summary,
)

# All data below is synthetic.
DAYS = pd.date_range("2020-01-01", periods=10, freq="D")


def labels_frame() -> pd.DataFrame:
    status = ["zone", "zone", "positive", "positive", "in_episode", "negative", "negative"]
    return pd.DataFrame(
        {
            "StationId": 1,
            "Day": DAYS[:7],  # days 8 to 10 have no label row
            "status": pd.Categorical(
                status, categories=["zone", "positive", "in_episode", "negative"]
            ),
            "split": "train",
            "label": pd.array([pd.NA, pd.NA, 1, 1, pd.NA, 0, 0], dtype="Int8"),
        }
    )


def values_frame() -> pd.DataFrame:
    v = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return pd.DataFrame({"StationId": 1, "Day": DAYS, "score": v})


def test_calendar_marks_unlabelled_days_and_keeps_values_on_dropped_days() -> None:
    cal = calendar_frame(values_frame(), labels_frame(), 1, "score")
    assert len(cal) == 10
    assert cal.loc["2020-01-01", "status"] == "zone" and cal.loc["2020-01-01", "value"] == 1.0
    assert cal.loc["2020-01-09", "status"] == NO_DATA and cal.loc["2020-01-09", "split"] == "none"
    assert cal.loc["2020-01-09", "value"] == 9.0  # a score exists even though the day has no label


def test_calendar_fills_days_missing_from_both_inputs() -> None:
    values = values_frame().drop(index=[4])
    labels = labels_frame().drop(index=[4])
    cal = calendar_frame(values, labels, 1, "score")
    assert len(cal) == 10  # 2020-01-05 is in neither input but stays on the calendar
    assert cal.loc["2020-01-05", "status"] == NO_DATA and pd.isna(cal.loc["2020-01-05", "value"])


def test_blank_reasons_and_summary_count_days_by_status() -> None:
    cal = calendar_frame(values_frame(), labels_frame(), 1, "score")
    reasons = blank_reasons(cal)
    assert reasons.to_dict() == {"positive": 1}  # only 2020-01-03 has no value
    summary = status_summary(cal)
    assert summary.loc["train", "zone"] == 2 and summary.loc["none", NO_DATA] == 3


def test_unknown_station_raises() -> None:
    with pytest.raises(ValueError):
        calendar_frame(values_frame(), labels_frame(), 99, "score")


def test_plot_builds_a_figure_with_episode_lines_and_four_axes() -> None:
    cal = calendar_frame(values_frame(), labels_frame(), 1, "score")
    episodes = pd.DataFrame(
        {
            "StationId": [1],
            "Start": [pd.Timestamp("2020-01-05")],
            "End": [pd.Timestamp("2020-01-05")],
        }
    )
    fig = plot_risk_timeline(cal, episodes, 1, ylabel="score", threshold=3.0)
    assert len(fig.axes) == 4
    legend_text = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "zone" in legend_text and "value missing" in legend_text
