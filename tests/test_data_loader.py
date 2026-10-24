import io

import pandas as pd
import pytest

from src.data_loader import load_market_data, normalize_column_name


def test_column_normalization_and_loading():
    csv = io.StringIO("Date,Close Price,Symbol\n2020-01-02,100,Sensex\n2020-01-03,101,Sensex\n")
    data, quality = load_market_data(csv)
    assert normalize_column_name("Close Price") == "closeprice"
    assert len(data) == 2
    assert quality.coverage_start == "2020-01-02"


def test_missing_required_columns():
    with pytest.raises(ValueError, match="date column"):
        load_market_data(io.StringIO("close\n10\n"))


def test_invalid_and_duplicate_rows_are_removed():
    csv = io.StringIO("date,close\n2020-01-02,100\n2020-01-02,100\nnot-a-date,50\n2020-01-03,-1\n")
    data, quality = load_market_data(csv)
    assert len(data) == 1
    assert quality.duplicate_rows == 1

