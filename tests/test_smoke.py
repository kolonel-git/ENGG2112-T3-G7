def test_import_package() -> None:
    import wt_predmaint
    from wt_predmaint import data, evaluation, explain, features, labels, models  # noqa: F401

    assert wt_predmaint.__version__
