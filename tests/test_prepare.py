import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.features.prepare import TrainOnlyPreprocessor


def make_train(*, index: list[int] | None = None, **features: list[object]) -> pd.DataFrame:
    """Synthetic labelled train rows with the supplied feature columns."""
    n_rows = len(next(iter(features.values())))
    frame = pd.DataFrame(
        {
            **features,
            "status": ["negative"] * n_rows,
            "label": np.arange(n_rows) % 2,
            "split": ["train"] * n_rows,
        },
        index=index,
    )
    return frame


def processor(
    columns: list[str],
    *,
    min_observed_share: float = 0.5,
    add_missing_indicators: bool = False,
    scale: bool = True,
    clip_quantiles: tuple[float, float] | None = None,
) -> TrainOnlyPreprocessor:
    return TrainOnlyPreprocessor(
        columns,
        min_observed_share=min_observed_share,
        add_missing_indicators=add_missing_indicators,
        scale=scale,
        clip_quantiles=clip_quantiles,
    )


def test_sparse_nonfinite_and_constant_columns_are_selected_from_train() -> None:
    train = make_train(
        keep=[1.0, 2.0, 3.0, 4.0],
        boundary=[1.0, 2.0, np.nan, np.nan],
        sparse=[1.0, np.nan, np.nan, np.nan],
        all_missing=[np.nan, np.nan, np.nan, np.nan],
        infinite_only=[np.inf, -np.inf, np.nan, np.nan],
        constant=[5.0, 5.0, np.nan, 5.0],
    )
    columns = ["keep", "boundary", "sparse", "all_missing", "infinite_only", "constant"]

    fitted = processor(columns).fit(train)

    assert fitted.observed_share_["boundary"] == 0.5  # cutoff is inclusive
    assert fitted.retained_columns_ == ["keep", "boundary"]
    assert fitted.dropped_sparse_ == ["sparse", "all_missing", "infinite_only"]
    assert fitted.dropped_constant_ == ["constant"]
    transformed = fitted.transform(train)
    assert transformed.columns.tolist() == ["keep", "boundary"]
    assert transformed.dtypes.tolist() == [np.dtype("float64"), np.dtype("float64")]
    assert np.isfinite(transformed.to_numpy()).all()


def test_imputation_and_scaling_use_frozen_train_statistics_and_are_row_independent() -> None:
    train = make_train(index=[10, 20, 30], x=[1.0, 3.0, np.nan], y=[2.0, 4.0, 6.0])
    before_train = train.copy(deep=True)
    fitted = processor(["x", "y"]).fit(train)

    assert fitted.medians_["x"] == 2.0
    assert fitted.means_["x"] == 2.0
    assert fitted.scales_["x"] == pytest.approx(np.sqrt(2.0 / 3.0))

    rows = pd.DataFrame(
        {
            "x": [np.nan, 100.0],
            "y": [8.0, 10.0],
            "label": [0, 1],
            "status": ["negative", "positive"],
            "split": ["val", "val"],
        },
        index=[101, 55],
    )
    before_rows = rows.copy(deep=True)
    full = fitted.transform(rows)

    assert full.index.equals(rows.index)
    assert full.loc[101, "x"] == 0.0
    assert full.loc[55, "x"] == pytest.approx((100.0 - 2.0) / np.sqrt(2.0 / 3.0))
    for row_index in rows.index:
        pd.testing.assert_frame_equal(
            full.loc[[row_index]], fitted.transform(rows.loc[[row_index]])
        )

    blind = rows.copy()
    blind["label"] = 1 - blind["label"]
    blind["status"] = "zone"
    pd.testing.assert_frame_equal(full, fitted.transform(blind))
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(rows, before_rows)


def test_scale_false_keeps_imputed_values_and_freezes_column_order() -> None:
    train = make_train(b=[10.0, 20.0, 30.0], a=[1.0, np.nan, 5.0])
    fitted = processor(["a", "b"], scale=False).fit(train)
    rows = pd.DataFrame(
        {"b": [40.0, 50.0], "extra": [999.0, 999.0], "a": [np.nan, 7.0]},
        index=[8, 3],
    )

    first = fitted.transform(rows)
    second = fitted.transform(rows)

    assert fitted.get_feature_names_out() == ["a", "b"]
    assert first.columns.tolist() == ["a", "b"]
    assert first.index.tolist() == [8, 3]
    assert first["a"].tolist() == [3.0, 7.0]
    assert first["b"].tolist() == [40.0, 50.0]
    pd.testing.assert_frame_equal(first, second)


def test_missing_indicators_are_frozen_from_train_and_remain_binary() -> None:
    train = make_train(x=[1.0, np.nan, 3.0], y=[10.0, 20.0, 30.0])
    fitted = processor(
        ["x", "y"], add_missing_indicators=True, scale=False
    ).fit(train)
    rows = pd.DataFrame({"x": [np.nan, 5.0], "y": [np.nan, 40.0]}, index=[4, 9])

    transformed = fitted.transform(rows)

    assert fitted.indicator_source_columns_ == ["x"]
    assert fitted.indicator_columns_ == ["x__missing"]
    assert transformed.columns.tolist() == ["x", "y", "x__missing"]
    assert transformed["x__missing"].tolist() == [1.0, 0.0]
    assert "y__missing" not in transformed
    assert transformed.loc[4, "y"] == 20.0


def test_optional_clipping_uses_train_quantiles_before_imputation() -> None:
    train = make_train(x=[0.0, 10.0, 20.0, 30.0])
    fitted = processor(
        ["x"], scale=False, clip_quantiles=(0.25, 0.75)
    ).fit(train)
    rows = pd.DataFrame({"x": [-999.0, 999.0, np.nan]})

    transformed = fitted.transform(rows)

    assert fitted.clip_lower_["x"] == 7.5
    assert fitted.clip_upper_["x"] == 22.5
    assert fitted.medians_["x"] == 15.0
    assert transformed["x"].tolist() == [7.5, 22.5, 15.0]


def test_changing_validation_or_test_rows_cannot_change_the_fit() -> None:
    train = make_train(x=[1.0, 2.0, 4.0], y=[10.0, np.nan, 30.0])
    later_a = pd.DataFrame(
        {"x": [5.0, 6.0], "y": [40.0, 50.0], "label": [0, 1], "split": ["val", "test"]}
    )
    later_b = later_a.assign(x=[-1e12, 1e12], y=[np.nan, np.inf])
    table_a = pd.concat([train, later_a], ignore_index=True)
    table_b = pd.concat([train, later_b], ignore_index=True)

    fitted_a = processor(["x", "y"], add_missing_indicators=True).fit(
        table_a.loc[table_a["split"] == "train"]
    )
    fitted_b = processor(["x", "y"], add_missing_indicators=True).fit(
        table_b.loc[table_b["split"] == "train"]
    )

    pd.testing.assert_series_equal(fitted_a.medians_, fitted_b.medians_)
    pd.testing.assert_series_equal(fitted_a.means_, fitted_b.means_)
    pd.testing.assert_series_equal(fitted_a.scales_, fitted_b.scales_)
    assert fitted_a.get_feature_names_out() == fitted_b.get_feature_names_out()


def test_fit_requires_nonempty_labelled_train_rows() -> None:
    validation = make_train(x=[1.0, 2.0]).assign(split="val")
    with pytest.raises(ValueError, match="outside the train split"):
        processor(["x"]).fit(validation)

    unlabelled = make_train(x=[1.0, 2.0])
    unlabelled.loc[unlabelled.index[0], "label"] = np.nan
    with pytest.raises(ValueError, match="without a label"):
        processor(["x"]).fit(unlabelled)

    empty = make_train(x=[])
    with pytest.raises(ValueError, match="empty training frame"):
        processor(["x"]).fit(empty)


def test_fit_rejects_nonnumeric_boolean_and_no_retained_features() -> None:
    nonnumeric = make_train(x=["1", "2"])
    with pytest.raises(TypeError, match="numeric and not boolean"):
        processor(["x"]).fit(nonnumeric)

    boolean = make_train(x=[True, False])
    with pytest.raises(TypeError, match="numeric and not boolean"):
        processor(["x"]).fit(boolean)

    constants = make_train(x=[1.0, 1.0], y=[np.nan, np.nan])
    with pytest.raises(ValueError, match="no feature columns remain"):
        processor(["x", "y"]).fit(constants)


def test_transform_requires_fit_and_every_fitted_source_column() -> None:
    unfitted = processor(["x"])
    with pytest.raises(RuntimeError, match="must be fitted"):
        unfitted.transform(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(RuntimeError, match="must be fitted"):
        unfitted.get_feature_names_out()

    fitted = processor(["x"]).fit(make_train(x=[1.0, 2.0]))
    with pytest.raises(KeyError, match="missing fitted feature columns"):
        fitted.transform(pd.DataFrame({"other": [1.0]}))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"columns": []}, ValueError, "at least one"),
        ({"columns": "x"}, TypeError, "not a string"),
        ({"columns": ["x", "x"]}, ValueError, "unique"),
        ({"columns": ["label"]}, ValueError, "metadata"),
        ({"columns": ["x"], "min_observed_share": 0.0}, ValueError, r"in \(0, 1\]"),
        ({"columns": ["x"], "min_observed_share": 1.1}, ValueError, r"in \(0, 1\]"),
        ({"columns": ["x"], "clip_quantiles": (0.8, 0.2)}, ValueError, "lower < upper"),
        ({"columns": ["x"], "clip_quantiles": [0.1, 0.9]}, TypeError, "tuple"),
    ],
)
def test_constructor_validation(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    options: dict[str, object] = {
        "columns": ["x"],
        "min_observed_share": 0.5,
        "add_missing_indicators": False,
    }
    options.update(kwargs)
    with pytest.raises(error, match=message):
        TrainOnlyPreprocessor(**options)  # type: ignore[arg-type]


def test_indicator_name_collisions_and_duplicate_frame_columns_are_rejected() -> None:
    collision = make_train(x=[1.0, np.nan, 3.0], x__missing=[4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="indicator names collide"):
        processor(["x", "x__missing"], add_missing_indicators=True).fit(collision)

    duplicate_frame = pd.DataFrame(
        [[1.0, 2.0, 0, "train"], [3.0, 4.0, 1, "train"]],
        columns=["x", "x", "label", "split"],
    )
    with pytest.raises(ValueError, match="exactly once"):
        processor(["x"]).fit(duplicate_frame)


def test_get_feature_names_returns_a_copy() -> None:
    fitted = processor(["x"]).fit(make_train(x=[1.0, 2.0]))
    names = fitted.get_feature_names_out()
    names.append("corruption")
    assert fitted.get_feature_names_out() == ["x"]
    assert copy.copy(fitted.get_feature_names_out()) == ["x"]
