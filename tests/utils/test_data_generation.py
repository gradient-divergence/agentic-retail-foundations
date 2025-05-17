import pandas as pd
import pytest
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

# Function to test
from utils.data_generation import generate_synthetic_retail_data

# Define common parameters for tests
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-01-07"  # 1 week
TEST_NUM_STORES = 2
TEST_NUM_PRODUCTS = 3
TEST_SEED = 123


@pytest.fixture(scope="module")  # Generate data once for the module
def generated_data():
    """Fixture to generate data once for multiple tests."""
    return generate_synthetic_retail_data(
        start_date_str=TEST_START_DATE,
        end_date_str=TEST_END_DATE,
        num_stores=TEST_NUM_STORES,
        num_products=TEST_NUM_PRODUCTS,
        seed=TEST_SEED,
    )


def test_output_types(generated_data):
    """Test that the function returns three pandas DataFrames."""
    sales_df, product_df, store_df = generated_data
    assert isinstance(sales_df, pd.DataFrame)
    assert isinstance(product_df, pd.DataFrame)
    assert isinstance(store_df, pd.DataFrame)


def test_output_dimensions(generated_data):
    """Test the dimensions of the output DataFrames."""
    sales_df, product_df, store_df = generated_data
    num_days = (pd.to_datetime(TEST_END_DATE) - pd.to_datetime(TEST_START_DATE)).days + 1

    assert len(store_df) == TEST_NUM_STORES
    assert len(product_df) == TEST_NUM_PRODUCTS
    assert len(sales_df) == num_days * TEST_NUM_STORES * TEST_NUM_PRODUCTS


def test_sales_df_columns_and_types(generated_data):
    """Test the columns and data types in the sales DataFrame."""
    sales_df, _, _ = generated_data
    expected_cols_types = {
        "date": is_datetime64_any_dtype,
        "store_id": is_object_dtype,  # String/Object
        "product_id": is_object_dtype,  # String/Object
        "sales_units": is_integer_dtype,  # Should be integer after Poisson
        "price": is_float_dtype,
        "on_promotion": is_bool_dtype,
        "store_traffic": is_integer_dtype,
        "product_category": is_object_dtype,
        "store_tier": is_object_dtype,
    }
    assert set(sales_df.columns) == set(expected_cols_types.keys())
    for col, type_check_func in expected_cols_types.items():
        assert type_check_func(sales_df[col]), f"Column '{col}' failed type check {type_check_func.__name__}"


def test_product_df_columns_and_types(generated_data):
    """Test the columns and data types in the product DataFrame."""
    _, product_df, _ = generated_data
    expected_cols_types = {
        "product_id": is_object_dtype,
        "product_category": is_object_dtype,
    }
    assert set(product_df.columns) == set(expected_cols_types.keys())
    for col, type_check_func in expected_cols_types.items():
        assert type_check_func(product_df[col]), f"Column '{col}' failed type check {type_check_func.__name__}"


def test_store_df_columns_and_types(generated_data):
    """Test the columns and data types in the store DataFrame."""
    _, _, store_df = generated_data
    expected_cols_types = {
        "store_id": is_object_dtype,
        "store_tier": is_object_dtype,
    }
    assert set(store_df.columns) == set(expected_cols_types.keys())
    for col, type_check_func in expected_cols_types.items():
        assert type_check_func(store_df[col]), f"Column '{col}' failed type check {type_check_func.__name__}"


def test_data_relationships_and_ranges(generated_data):
    """Test relationships between tables and basic value ranges."""
    sales_df, product_df, store_df = generated_data

    # --- Test Ranges --- #
    assert (sales_df["sales_units"] >= 0).all()
    assert (sales_df["price"] > 0).all()
    assert (sales_df["store_traffic"] >= 0).all()
    assert sales_df["on_promotion"].isin([True, False]).all()

    # --- Test Relationships --- #
    # Check foreign key like relationships
    assert set(sales_df["store_id"].unique()).issubset(set(store_df["store_id"].unique()))
    assert set(sales_df["product_id"].unique()).issubset(set(product_df["product_id"].unique()))

    # Check merged data consistency
    merged_store = pd.merge(sales_df, store_df, on="store_id", how="left", suffixes=("", "_store"))
    assert (merged_store["store_tier"] == merged_store["store_tier_store"]).all()

    merged_product = pd.merge(sales_df, product_df, on="product_id", how="left", suffixes=("", "_prod"))
    assert (merged_product["product_category"] == merged_product["product_category_prod"]).all()


def test_reproducibility_with_seed():
    """Test that using the same seed produces identical results."""
    params = {
        "start_date_str": TEST_START_DATE,
        "end_date_str": TEST_END_DATE,
        "num_stores": TEST_NUM_STORES,
        "num_products": TEST_NUM_PRODUCTS,
        # Use a specific seed for this test
        "seed": 42,
    }

    # Generate data first time
    sales_df1, product_df1, store_df1 = generate_synthetic_retail_data(**params)

    # Generate data second time with the same seed
    sales_df2, product_df2, store_df2 = generate_synthetic_retail_data(**params)

    # Assert DataFrames are identical
    pd.testing.assert_frame_equal(sales_df1, sales_df2)
    pd.testing.assert_frame_equal(product_df1, product_df2)
    pd.testing.assert_frame_equal(store_df1, store_df2)

    # Generate data third time with a different seed
    params["seed"] = 43
    sales_df3, product_df3, store_df3 = generate_synthetic_retail_data(**params)

    # Assert DataFrames are different from the first run
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(sales_df1, sales_df3)
    # Product/Store DFs might be identical if only seed changes internal sales
    # randomness. Let's check just sales_df for difference
    # with pytest.raises(AssertionError):
    #     pd.testing.assert_frame_equal(product_df1, product_df3)


# Placeholder for spot checks
# def test_spot_check_promotion_effect(): ...
