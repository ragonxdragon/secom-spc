from secom_spc import load_secom


def test_load_secom_parses_labels_sorts_by_time(tmp_path):
    (tmp_path / "secom.data").write_text("1 2\n3 NaN\n5 6\n")
    (tmp_path / "secom_labels.data").write_text(
        '-1 "02/01/2008 10:00:00"\n1 "01/01/2008 09:30:00"\n-1 "03/01/2008 00:00:00"\n'
    )
    x, y = load_secom(tmp_path)
    assert list(y) == [1, 0, 0]
    assert x.index.is_monotonic_increasing
    assert x.index[0].day == 1
    assert x.shape == (3, 2)
    assert x["s001"].isna().sum() == 1
