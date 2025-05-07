"""Functions for calculating and interpreting Return on Investment (ROI) based on causal impact estimates."""

import numpy as np
from typing import Any, Dict, Optional

def calculate_promotion_roi(
    estimated_ate: Optional[float],
    average_baseline_sales: float,
    num_treated_units: int,
    promotion_cost_per_instance: float,
    margin_percent: float,
    treatment_variable: str = "promotion_applied",
) -> Optional[Dict[str, Any]]:
    """
    Calculates the Return on Investment (ROI) for a promotion based on the estimated ATE.

    Args:
        estimated_ate: The estimated Average Treatment Effect (uplift in sales per treated unit).
        average_baseline_sales: The average sales value per unit in the absence of treatment.
        num_treated_units: The number of units (e.g., customer-product-day) that received the promotion.
        promotion_cost_per_instance: The cost incurred for each instance of the promotion being applied.
        margin_percent: The profit margin on sales (e.g., 0.2 for 20%).
        treatment_variable: Name of the treatment variable (for reporting).

    Returns:
        A dictionary containing ROI calculations (total uplift, total cost, net profit, ROI),
        or None if ATE is None.
    """
    if estimated_ate is None:
        print("Cannot calculate ROI because Estimated ATE is None.")
        return None

    if num_treated_units <= 0:
        print("Warning: Number of treated units is zero or negative. ROI cannot be calculated.")
        return {
            "error": "Number of treated units is non-positive.",
            "estimated_ate": estimated_ate,
            "num_treated_units": num_treated_units,
            # Include other inputs for context
        }

    # Calculate total estimated sales uplift due to the promotion
    total_sales_uplift = estimated_ate * num_treated_units

    # Calculate total cost of the promotion
    total_promotion_cost = promotion_cost_per_instance * num_treated_units

    # Calculate the profit generated from the uplift
    profit_from_uplift = total_sales_uplift * margin_percent

    # Calculate net profit (or loss) from the promotion
    net_profit = profit_from_uplift - total_promotion_cost

    # Calculate ROI
    # Avoid division by zero if total cost is zero
    if total_promotion_cost > 0:
        roi_percentage = (net_profit / total_promotion_cost) * 100
    elif net_profit > 0:
        roi_percentage = np.inf # Positive profit with zero cost -> infinite ROI
    else:
        roi_percentage = 0 # Zero or negative profit with zero cost -> 0 ROI

    print("\nPromotion ROI Calculation:")
    print(f"  Estimated ATE ({treatment_variable}): {estimated_ate:.4f}")
    print(f"  Number of Treated Units: {num_treated_units}")
    print(f"  Average Baseline Sales (per unit): {average_baseline_sales:.2f}")
    print(f"  Promotion Cost per Instance: {promotion_cost_per_instance:.2f}")
    print(f"  Profit Margin: {margin_percent:.1%}")
    print(f"  ----------------------------------------")
    print(f"  Total Estimated Sales Uplift: {total_sales_uplift:.2f}")
    print(f"  Profit from Uplift: {profit_from_uplift:.2f}")
    print(f"  Total Promotion Cost: {total_promotion_cost:.2f}")
    print(f"  Net Profit from Promotion: {net_profit:.2f}")
    print(f"  Return on Investment (ROI): {roi_percentage:.2f}%")

    return {
        "estimated_ate": estimated_ate,
        "num_treated_units": num_treated_units,
        "average_baseline_sales": average_baseline_sales,
        "promotion_cost_per_instance": promotion_cost_per_instance,
        "margin_percent": margin_percent,
        "total_sales_uplift": total_sales_uplift,
        "profit_from_uplift": profit_from_uplift,
        "total_promotion_cost": total_promotion_cost,
        "net_profit": net_profit,
        "roi_percentage": roi_percentage,
    }

def interpret_causal_impact(roi_results: Optional[Dict[str, Any]]) -> str:
    """
    Provides a simple textual interpretation of the ROI results.

    Args:
        roi_results: The dictionary returned by calculate_promotion_roi.

    Returns:
        A string summarizing the financial impact of the promotion.
    """
    if roi_results is None or "error" in roi_results:
        return "ROI calculation could not be completed or resulted in an error."

    ate = roi_results["estimated_ate"]
    net_profit = roi_results["net_profit"]
    roi = roi_results["roi_percentage"]

    interpretation = f"The analysis estimated an Average Treatment Effect (ATE) of {ate:.4f}. "

    if net_profit > 0:
        interpretation += (
            f"This translates to a positive financial impact, with an estimated net profit of "
            f"${net_profit:,.2f} and a Return on Investment (ROI) of {roi:.2f}%."
        )
    elif net_profit == 0:
        interpretation += (
            f"This translates to a break-even scenario, with an estimated net profit of "
            f"${net_profit:,.2f} and an ROI of {roi:.2f}%."
        )
    else:
        interpretation += (
            f"This translates to a negative financial impact, with an estimated net loss of "
            f"${abs(net_profit):,.2f} (ROI: {roi:.2f}%). The promotion costs outweighed the profit generated from the sales uplift."
        )

    # Add considerations based on ATE significance if available (e.g., from p-value)
    # This would require passing p-value information into the ROI calculation or interpretation step.
    # Example: if p_value is not None and p_value < 0.05:
    #    interpretation += " The estimated ATE is statistically significant at the 5% level."
    # else:
    #    interpretation += " Caution: The estimated ATE may not be statistically significant."

    print(f"\nInterpretation: {interpretation}")
    return interpretation 