# tests/agents/causal_analysis/test_data_preparation.py

import numpy as np
import pandas as pd
import pytest

# Assuming prepare_analysis_data is accessible via this import path
# Adjust if the project structure requires a different import
from agents.causal_analysis.data_preparation import prepare_analysis_data

# --- Fixtures for Test Data ---


@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    """Basic sales data fixture."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"]),  # Day 3 has NaN sales
            "store_id": [1, 2, 1, 2, 1],
            "product_id": [101, 101, 102, 101, 101],
            "sales": [10.0, 12.0, 5.0, 8.0, np.nan],
            "price": [1.0, 1.0, 2.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def sample_product_data() -> pd.DataFrame:
    """Basic product data fixture."""
    return pd.DataFrame(
        {
            "product_id": [101, 102, 103],  # Product 103 is extra
            "category": ["A", "B", "A"],
            "brand": ["X", "Y", "X"],
        }
    )


@pytest.fixture
def sample_store_data() -> pd.DataFrame:
    """Basic store data fixture."""
    return pd.DataFrame(
        {
            "store_id": [1, 2],
            "location": ["Urban", "Suburban"],
            "size_sqft": [1000, 1500],
        }
    )


@pytest.fixture
def sample_promotion_data() -> pd.DataFrame:
    """Basic promotion data fixture."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),  # Promo on day 1 & 2
            "store_id": [1, 1],  # Only store 1
            "product_id": [101, 102],  # Product 101 (day 1), 102 (day 2)
            "promotion_applied": [1, 1],
            "promo_type": ["Discount", "BOGO"],
        }
    )


@pytest.fixture
def sample_promotion_data_missing_col() -> pd.DataFrame:
    """Promotion data missing the required 'promotion_applied' column."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01"]),
            "store_id": [1],
            "product_id": [101],
            "promo_type": ["Discount"],
        }
    )


@pytest.fixture
def sample_promotion_data_no_match() -> pd.DataFrame:
    """Promotion data with keys that won't match sample_sales_data."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-10", "2023-01-11"]),  # Dates outside sales data range
            "store_id": [99, 99],
            "product_id": [999, 999],
            "promotion_applied": [1, 1],
            "promo_type": ["Discount", "BOGO"],
        }
    )


# --- Test Cases ---


def test_prepare_all_data_present(sample_sales_data, sample_product_data, sample_store_data, sample_promotion_data):
    """Test successful preparation when all dataframes are provided."""
    result_df = prepare_analysis_data(
        sales_data=sample_sales_data,
        product_data=sample_product_data,
        store_data=sample_store_data,
        promotion_data=sample_promotion_data,
    )

    # Expected shape: 5 initial rows, 1 row dropped due to NaN sales -> 4 rows
    assert result_df.shape[0] == 4
    # Expected columns: sales cols + product cols + store cols + promo col + time feats
    expected_cols = set(
        [
            "date",
            "store_id",
            "product_id",
            "sales",
            "price",  # from sales
            "category",
            "brand",  # from product
            "location",
            "size_sqft",  # from store
            "promotion_applied",  # from promo (promo_type dropped implicitly)
            "day_of_week",
            "month",
            "year",
        ]
    )  # generated
    assert set(result_df.columns) == expected_cols

    # Check promotion merge logic
    # Row 0: date=01-01, store=1, prod=101 -> promo=1
    assert result_df.iloc[0]["promotion_applied"] == 1
    # Row 1: date=01-01, store=2, prod=101 -> no promo match -> promo=0
    assert result_df.iloc[1]["promotion_applied"] == 0
    # Row 2: date=01-02, store=1, prod=102 -> promo=1
    assert result_df.iloc[2]["promotion_applied"] == 1
    # Row 3: date=01-02, store=2, prod=101 -> no promo match -> promo=0
    assert result_df.iloc[3]["promotion_applied"] == 0

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
    assert pd.api.types.is_integer_dtype(result_df["promotion_applied"])
    assert pd.api.types.is_integer_dtype(result_df["day_of_week"])
    assert pd.api.types.is_integer_dtype(result_df["month"])
    assert pd.api.types.is_integer_dtype(result_df["year"])

    # Check NaN drop
    assert not result_df["sales"].isnull().any()


def test_prepare_only_sales_data(sample_sales_data):
    """Test preparation with only the mandatory sales data."""
    result_df = prepare_analysis_data(sales_data=sample_sales_data)

    assert result_df.shape[0] == 4  # Still drops NaN row
    expected_cols = set(
        [
            "date",
            "store_id",
            "product_id",
            "sales",
            "price",  # from sales
            "promotion_applied",  # added by default
            "day_of_week",
            "month",
            "year",
        ]
    )  # generated
    assert set(result_df.columns) == expected_cols
    assert (result_df["promotion_applied"] == 0).all()  # Should be 0 everywhere


def test_prepare_sales_and_promotions(sample_sales_data, sample_promotion_data):
    """Test preparation with sales and promotions, but no product/store info."""
    result_df = prepare_analysis_data(sales_data=sample_sales_data, promotion_data=sample_promotion_data)
    assert result_df.shape[0] == 4
    expected_cols = set(
        [
            "date",
            "store_id",
            "product_id",
            "sales",
            "price",  # from sales
            "promotion_applied",  # from promo
            "day_of_week",
            "month",
            "year",
        ]
    )  # generated
    assert set(result_df.columns) == expected_cols
    # Check promotions are merged correctly
    assert result_df.iloc[0]["promotion_applied"] == 1
    assert result_df.iloc[1]["promotion_applied"] == 0
    assert result_df.iloc[2]["promotion_applied"] == 1
    assert result_df.iloc[3]["promotion_applied"] == 0


def test_missing_required_sales_column(sample_sales_data):
    """Test ValueError if 'sales' column is missing."""
    bad_sales_data = sample_sales_data.drop(columns=["sales"])
    with pytest.raises(ValueError, match="Sales data must contain a 'sales' column."):
        prepare_analysis_data(sales_data=bad_sales_data)


def test_missing_required_promotion_column(sample_sales_data, sample_promotion_data_missing_col):
    """Test ValueError if 'promotion_applied' is missing in promotion_data."""
    with pytest.raises(
        ValueError,
        match="Promotion data is missing required columns: .*'promotion_applied'",
    ):
        prepare_analysis_data(
            sales_data=sample_sales_data,
            promotion_data=sample_promotion_data_missing_col,
        )


def test_missing_optional_product_id(sample_sales_data, sample_product_data):
    """Test ValueError if product_data is provided but lacks 'product_id'."""
    bad_product_data = sample_product_data.drop(columns=["product_id"])
    with pytest.raises(ValueError, match="Product data must contain 'product_id' column."):
        prepare_analysis_data(sales_data=sample_sales_data, product_data=bad_product_data)


def test_missing_optional_store_id(sample_sales_data, sample_store_data):
    """Test ValueError if store_data is provided but lacks 'store_id'."""
    bad_store_data = sample_store_data.drop(columns=["store_id"])
    with pytest.raises(ValueError, match="Store data must contain 'store_id' column."):
        prepare_analysis_data(sales_data=sample_sales_data, store_data=bad_store_data)


def test_date_conversion_string_input(sample_sales_data):
    """Test that string dates are converted correctly."""
    sales_str_date = sample_sales_data.copy()
    sales_str_date["date"] = sales_str_date["date"].dt.strftime("%Y-%m-%d")
    result_df = prepare_analysis_data(sales_data=sales_str_date)
    assert pd.api.types.is_datetime64_any_dtype(result_df["date"])
    assert result_df.shape[0] == 4  # Check NaN drop still works


def test_no_matching_promo_values(sample_sales_data, sample_promotion_data_no_match):
    """Test case where promotion data keys exist but values don't match sales data."""
    # This scenario should *not* raise a ValueError or a Warning.
    # It should simply result in no promotions being merged.
    result_df = prepare_analysis_data(sales_data=sample_sales_data, promotion_data=sample_promotion_data_no_match)

    # Check the output: promotion_applied should be all 0s because no merge occurred
    assert result_df.shape[0] == 4  # Ensure NaN drop still happened
    assert "promotion_applied" in result_df.columns
    assert (result_df["promotion_applied"] == 0).all()
    assert pd.api.types.is_integer_dtype(result_df["promotion_applied"])  # Check default type


def test_prepare_data_with_nan_promotions(sample_sales_data, sample_promotion_data):
    """Test handling when promotion merge introduces NaNs (should become 0)."""
    # Modify sales data to include a row that *won't* match the promo data
    extra_row = pd.DataFrame(
        {
            "date": [pd.to_datetime("2023-01-01")],
            "store_id": [3],  # Non-existent store in promo data
            "product_id": [101],
            "sales": [15.0],
            "price": [1.0],
        }
    )
    extended_sales_data = pd.concat([sample_sales_data, extra_row], ignore_index=True)

    result_df = prepare_analysis_data(sales_data=extended_sales_data, promotion_data=sample_promotion_data)

    # Expected rows: 5 original valid sales rows + 1 extra = 6 -> 1 dropped for NaN sales -> 5 rows
    assert result_df.shape[0] == 5

    # Check the promotion column specifically for the row that didn't match
    # The added row (index 4 after NaN drop) should have promotion_applied = 0
    assert result_df.loc[result_df["store_id"] == 3, "promotion_applied"].iloc[0] == 0
    # Check the original matched row still has 1
    assert (
        result_df.loc[
            (result_df["date"] == pd.to_datetime("2023-01-01")) & (result_df["store_id"] == 1),
            "promotion_applied",
        ].iloc[0]
        == 1
    )
    # Ensure dtype is still integer
    assert pd.api.types.is_integer_dtype(result_df["promotion_applied"])
