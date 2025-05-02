import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_retail_data(
    start_date_str: str = "2023-01-01",
    end_date_str: str = "2023-06-30",
    num_stores: int = 5,
    num_products: int = 20,
    seed: int = 42,
    base_traffic_lambda: int = 100,
    weekend_traffic_multiplier: float = 1.3,
    store_tier_effect_base: float = 0.1,
    product_base_sales_lambda_base: int = 10,
    product_category_sales_mult: int = 2,
    product_base_price_start: float = 9.99,
    product_category_price_add: int = 5,
    product_id_price_add: float = 0.5,
    seasonal_effect_amplitude: float = 0.2,
    weekend_sales_effect: float = 0.15,
    promo_base_prob: float = 0.05,
    promo_weekend_add_prob: float = 0.15,
    promo_cat2_add_prob: float = 0.05, # Example: Cat2 more likely promoted
    promo_store1_add_prob: float = 0.05, # Example: Store S01 more likely promotes
    true_promo_effect_multiplier: float = 1.5,
    promo_price_discount: float = 0.20,
    noise_std_dev: float = 0.1,
    traffic_sales_effect_divisor: float = 2000.0,
    store_traffic_noise_std_dev: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic retail sales data with confounding factors.

    Args:
        start_date_str: Start date string (YYYY-MM-DD).
        end_date_str: End date string (YYYY-MM-DD).
        num_stores: Number of stores to simulate.
        num_products: Number of products to simulate.
        seed: Random seed for reproducibility.
        base_traffic_lambda: Avg daily traffic (Poisson mean).
        weekend_traffic_multiplier: Multiplier for weekend traffic.
        store_tier_effect_base: Additive effect per store tier.
        product_base_sales_lambda_base: Base daily sales (Poisson mean).
        product_category_sales_mult: Additive sales per category number.
        product_base_price_start: Base price for Cat1, Prod 1.
        product_category_price_add: Added price per category number.
        product_id_price_add: Small added price variation per product ID.
        seasonal_effect_amplitude: Amplitude of sine wave for monthly seasonality.
        weekend_sales_effect: Multiplicative effect on sales for weekends.
        promo_base_prob: Base probability of an item being on promotion.
        promo_weekend_add_prob: Added probability on weekends.
        promo_cat2_add_prob: Added probability for category 2.
        promo_store1_add_prob: Added probability for store S01.
        true_promo_effect_multiplier: The actual causal lift from promotion.
        promo_price_discount: Price reduction when on promotion (e.g., 0.20 for 20%).
        noise_std_dev: Standard deviation of multiplicative normal noise on sales.
        traffic_sales_effect_divisor: Larger values mean smaller traffic effect on sales.
        store_traffic_noise_std_dev: Std dev of noise added to daily store traffic.


    Returns:
        A tuple containing:
        - sales_df: DataFrame with daily sales transactions.
        - product_df: DataFrame with product metadata.
        - store_df: DataFrame with store metadata.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date_str, end=end_date_str)
    stores = [f"S{i:02d}" for i in range(1, num_stores + 1)]
    products = [f"P{i:03d}" for i in range(1, num_products + 1)]

    data = []
    print(
        f"Generating data for {len(dates)} dates, {len(stores)} stores, {len(products)} products..."
    )
    for date in dates:
        day_of_week = date.dayofweek
        month = date.month
        is_weekend = day_of_week >= 5
        # Simulate overall store traffic (higher on weekends, slight month trend)
        base_traffic = np.random.poisson(base_traffic_lambda) + month * 5
        store_traffic_day = base_traffic * (1 + (weekend_traffic_multiplier - 1) * is_weekend)

        for store_id in stores:
            store_num = int(store_id[1:])
            # Assign tiers 0, 1, 2 based on store number
            store_tier = store_num % 3
            store_tier_effect = 1.0 + store_tier * store_tier_effect_base

            for product_id in products:
                prod_num = int(product_id[1:])
                # ~4 product categories, ensures robustness if num_products < 4
                products_per_cat = max(1, num_products // 4)
                cat_num = (prod_num - 1) // products_per_cat + 1
                product_category = f"Cat{cat_num}"

                # Baseline sales and price influenced by category and product ID variation
                product_base_sales = np.random.poisson(
                    product_base_sales_lambda_base + cat_num * product_category_sales_mult
                )
                product_base_price = (
                    product_base_price_start
                    + cat_num * product_category_price_add
                    + (prod_num % 5) * product_id_price_add # Small variation within category
                )

                # Time effects on sales (seasonal sine wave + weekend boost)
                month_effect = (
                    1.0 + np.sin((month - 1) / 12 * 2 * np.pi) * seasonal_effect_amplitude
                )
                dow_effect = 1.0 + weekend_sales_effect * is_weekend

                # Promotion Decision Logic (Confounding!) - depends on weekend, category, store
                promo_prob = promo_base_prob
                promo_prob += promo_weekend_add_prob * is_weekend
                promo_prob += promo_cat2_add_prob * (cat_num == 2) # Example: Cat 2 bias
                promo_prob += promo_store1_add_prob * (store_id == "S01") # Example: Store S01 bias
                # Ensure probability is valid
                promo_prob = np.clip(promo_prob, 0.0, 1.0)
                on_promotion = np.random.binomial(1, promo_prob)

                # Price is affected by promotion
                price = product_base_price * (1 - promo_price_discount * on_promotion)

                # Multiplicative noise factor
                noise = np.random.normal(1.0, noise_std_dev)

                # Calculate final sales (combining base, effects, noise)
                sales = (
                    product_base_sales
                    * store_tier_effect
                    * month_effect
                    * dow_effect
                    * noise
                )

                # Apply the TRUE causal effect of promotion
                if on_promotion:
                    sales *= true_promo_effect_multiplier

                # Add effect of store traffic (higher traffic slightly boosts sales)
                traffic_effect = 1.0 + (store_traffic_day / traffic_sales_effect_divisor)
                sales *= traffic_effect

                # Ensure sales are non-negative and use Poisson distribution for final count
                # Adding a small base (0.1) avoids issues with Poisson mean being zero
                final_sales_units = max(0, np.random.poisson(max(0.1, sales)))

                # Add noisy store traffic for the record
                noisy_store_traffic = int(
                    max(0, store_traffic_day * (1 + np.random.normal(0, store_traffic_noise_std_dev)))
                )

                data.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "sales_units": final_sales_units,
                        "price": round(price, 2),
                        "on_promotion": bool(on_promotion), # Keep as bool for clarity
                        "store_traffic": noisy_store_traffic,
                        "product_category": product_category,
                        # Store tier is added via merge later
                    }
                )

    sales_df = pd.DataFrame(data)

    # --- Create metadata dataframes --- 
    product_ids = [f"P{i:03d}" for i in range(1, num_products + 1)]
    product_df = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_category": [
                f"Cat{(int(p[1:]) - 1) // max(1, num_products // 4) + 1}" for p in product_ids
            ],
            # Add other product attributes if needed, e.g., base_price
            # "base_price": [product_base_price_start + ... for p in product_ids]
        }
    )

    store_ids = [f"S{i:02d}" for i in range(1, num_stores + 1)]
    store_df = pd.DataFrame(
        {
            "store_id": store_ids,
            "store_tier": [
                f"Tier{(int(s[1:]) % 3)}" for s in store_ids
            ],
            # Add other store attributes if needed, e.g., region, size
        }
    )

    # Add store tier to sales_df using a merge for consistency
    # Ensure 'store_tier' column exists before merge if sales_df could be empty
    if not sales_df.empty:
        sales_df = pd.merge(sales_df, store_df[['store_id', 'store_tier']], on="store_id", how="left")
    else:
        sales_df['store_tier'] = pd.Series(dtype='object') # Add empty column if no data


    print(f"Generated {len(sales_df)} sample sales records.")
    return sales_df, product_df, store_df 