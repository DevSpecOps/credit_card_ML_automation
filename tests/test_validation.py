import pytest
from pandera.errors import SchemaErrors

from src.data.validation import validate_raw


def test_validation_passes_on_valid(raw_df):
    validate_raw(raw_df)


def test_validation_fails_on_negative_limit(raw_df):
    bad_df = raw_df.copy()
    bad_df.loc[0, "LIMIT_BAL"] = -1000
    with pytest.raises(SchemaErrors):
        validate_raw(bad_df)


def test_validation_fails_on_invalid_target(raw_df):
    bad_df = raw_df.copy()
    bad_df.loc[0, "default_payment_next_month"] = 5
    with pytest.raises(SchemaErrors):
        validate_raw(bad_df)


def test_validation_fails_on_null(raw_df):
    bad_df = raw_df.copy()
    bad_df.loc[0, "AGE"] = None
    with pytest.raises(SchemaErrors):
        validate_raw(bad_df)


def test_validation_fails_on_missing_column(raw_df):
    bad_df = raw_df.drop(columns=["AGE"])
    with pytest.raises(Exception):
        validate_raw(bad_df)
