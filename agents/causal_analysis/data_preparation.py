import warnings

import pandas as pd


def prepare_analysis_data(
    sales_data: pd.DataFrame,
    product_data: pd.DataFrame | None = None,
    store_data: pd.DataFrame | None = None,
    promotion_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merges sales, product, store, and promotion data into a single DataFrame
    suitable for causal analysis.

    Args:
        sales_data: DataFrame with sales transactions (must include 'sales', 'product_id', 'store_id', 'date').
        product_data: Optional DataFrame with product features (must include 'product_id').
        store_data: Optional DataFrame with store features (must include 'store_id').
        promotion_data: Optional DataFrame with promotion details (must include 'promotion_applied', 'date', 'product_id', 'store_id').

    Returns:
        pd.DataFrame: The merged and prepared analysis dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    # Start with sales data
    analysis_df = sales_data.copy()

    # Convert date columns to datetime if they aren't already
    if "date" in analysis_df.columns and not pd.api.types.is_datetime64_any_dtype(analysis_df["date"]):
        analysis_df["date"] = pd.to_datetime(analysis_df["date"])

    # Merge product data if available
    if product_data is not None:
        if "product_id" not in product_data.columns:
            raise ValueError("Product data must contain 'product_id' column.")
        analysis_df = pd.merge(analysis_df, product_data, on="product_id", how="left")

    # Merge store data if available
    if store_data is not None:
        if "store_id" not in store_data.columns:
            raise ValueError("Store data must contain 'store_id' column.")
        analysis_df = pd.merge(analysis_df, store_data, on="store_id", how="left")

    # Merge promotion data if available
    if promotion_data is not None:
        # Ensure promotion data date is datetime
        if "date" in promotion_data.columns and not pd.api.types.is_datetime64_any_dtype(promotion_data["date"]):
            promotion_data["date"] = pd.to_datetime(promotion_data["date"])

        # Define required keys for merging promotions
        required_promo_keys = ["date", "product_id", "store_id", "promotion_applied"]
        missing_promo_keys = [k for k in required_promo_keys if k not in promotion_data.columns]
        if missing_promo_keys:
            raise ValueError(f"Promotion data is missing required columns: {missing_promo_keys}")

        merge_keys = ["date", "product_id", "store_id"]
        valid_merge_keys = [k for k in merge_keys if k in analysis_df.columns and k in promotion_data.columns]

        if not valid_merge_keys:
            warnings.warn(
                "Could not find common keys (date, product_id, store_id) to merge promotion data. Assuming no promotions applied.",
                UserWarning,
            )
            analysis_df["promotion_applied"] = 0
        else:
            # Select only relevant columns from promotion data before merge
            promo_cols_to_merge = valid_merge_keys + ["promotion_applied"]
            analysis_df = pd.merge(
                analysis_df,
                promotion_data[promo_cols_to_merge],
                on=valid_merge_keys,
                how="left",
            )
            # Fill NaNs from the merge with 0 (indicating no promotion)
            analysis_df["promotion_applied"] = analysis_df["promotion_applied"].fillna(0).astype(int)
    else:
        # If no promotion data provided, assume no promotions applied
        analysis_df["promotion_applied"] = 0

    # Feature Engineering (Example: Extract time features)
    if "date" in analysis_df.columns:
        analysis_df["day_of_week"] = analysis_df["date"].dt.dayofweek
        analysis_df["month"] = analysis_df["date"].dt.month
        analysis_df["year"] = analysis_df["date"].dt.year
        # Drop original date column if no longer needed for direct modeling? Consider keeping it.
        # analysis_df = analysis_df.drop(columns=['date'])

    # Handle missing values (example: simple imputation or dropping)
    # This needs careful consideration based on the specific dataset.
    # For now, only drop rows where outcome ('sales') or treatment ('promotion_applied') is missing.
    if "sales" not in analysis_df.columns:
        raise ValueError("Sales data must contain a 'sales' column.")
    analysis_df = analysis_df.dropna(subset=["sales", "promotion_applied"])

    # Potential future step: Convert categorical features to numerical
    # Example:
    # analysis_df = pd.get_dummies(analysis_df, columns=['category', 'location_type'], drop_first=True)

    print(f"Prepared analysis data with {analysis_df.shape[0]} rows and {analysis_df.shape[1]} columns.")
    # print("Columns:", analysis_df.columns.tolist()) # Keep commented out for cleaner logs

    return analysis_df


# Example Usage (can be removed or kept under if __name__ == '__main__'):
# if __name__ == '__main__':
#     # Create dummy dataframes
#     sales = pd.DataFrame({
#         'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
#         'store_id': [1, 2, 1, 2],
#         'product_id': [101, 101, 102, 101],
#         'sales': [10, 12, 5, 8]
#     })
#     products = pd.DataFrame({'product_id': [101, 102], 'category': ['A', 'B']})
#     stores = pd.DataFrame({'store_id': [1, 2], 'location': ['Urban', 'Suburban']})
#     promotions = pd.DataFrame({
#         'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
#         'store_id': [1, 2],
#         'product_id': [101, 101],
#         'promotion_applied': [1, 1]
#     })
#
#     try:
#         analysis_data = prepare_analysis_data(sales, products, stores, promotions)
#         print("\nAnalysis Data Head:")
#         print(analysis_data.head())
#         print("\nAnalysis Data Info:")
#         analysis_data.info()
#     except ValueError as e:
#         print(f"Error preparing data: {e}")
