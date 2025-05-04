import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def __cell1():
    # Set matplotlib to use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    import pandas as pd
    import numpy as np
    import pathlib
    import marimo as mo

    return mo, np, pathlib, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 7: Sensor Networks and Cognitive Systems

        Welcome to the "nervous system" of retail operations. This chapter covers 
        how intelligent environmental monitoring, sensor data fusion, and 
        cognitive decision-making integrate to create responsive, real-time 
        retail environments. You'll gain the technical proficiency to deploy 
        these powerful tools practically.

        The focus here is on how an agent system processes multi-source sensor 
        data to maintain real-time inventory awareness
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Processing Sensor Data for Rel-Time Agent Decision""")
    return


@app.cell
def __cell2(mo, pathlib):
    sensor_data_processing_path = pathlib.Path("assets/sensor-data-processing.svg")
    mo.image(src=sensor_data_processing_path)
    return


@app.cell
def _():

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Knowledge Graph for Retail Product Relationships


        This implementation demonstrates key patterns for integrating sensor data in retail:

        1. **Multi-source data ingestion** through both real-time (WebSockets) and batch (REST) APIs.
        2. **Source-specific processing** that handles the unique characteristics of each sensor type.
        3. **Confidence scoring** to account for varying reliability across sensor technologies.
        4. **Discrepancy tracking** that accumulates evidence before triggering operational responses.
        5. **Cross-validation** between complementary sensor inputs to increase accuracy.


        The following example demonstrates how to build, query, and reason with a retail knowledge graph:
        """
    )
    return


@app.cell
def _(mo, pathlib):
    # Define the path relative to the project root
    knowledge_graph_path = pathlib.Path(
        "assets/knowledge-graph-for-retail-product-relationships.svg"
    )

    # Display the image
    mo.image(src=knowledge_graph_path)
    return


@app.cell
def __cell4():

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Causal Inference for Promotion Effectiveness


        Causal inference is a critical methodology for discovering true cause-and-effect relationships in retail data. Causal frameworks elevate decision-making beyond correlation analysis, enabling retail agents to identify what truly drives consumer behavior and business outcomes. 

        In modern retail environments, sophisticated decision-making requires moving beyond merely identifying patterns and correlations in data. Retail agents must delve deeper to understand the reasons behind certain outcomes—why specific events occur, what directly influences customer behaviors, and how different actions might impact future performance. 

        This advanced capability, known as causal reasoning and counterfactual analysis, enables retail organizations to implement proactive strategies rather than reactive responses, significantly enhancing decision quality and business outcomes.

        The following example demonstrates how to apply causal inference techniques to measure true promotion effectiveness:
        """
    )
    return


@app.cell
def _(mo, pathlib):
    causal_inference_path = pathlib.Path(
        "assets/causal-inference-in-promotion-effectiveness.svg"
    )
    mo.image(src=causal_inference_path)
    return


@app.cell
def __cell6():
    from agents.promotion_causal import PromotionCausalAnalyzer

    return (PromotionCausalAnalyzer,)


@app.cell
def _(mo):
    mo.md(r"""### Configuration""")
    return


@app.cell
def _():
    # Data Generation Params
    START_DATE = "2023-01-01"
    END_DATE = "2023-06-30"
    NUM_STORES = 5
    NUM_PRODUCTS = 20
    RANDOM_SEED = 42

    # Analysis Params
    MATCHING_CALIPER = 0.05
    MATCHING_RATIO = 1
    ANALYSES_TO_RUN = [
        "naive",
        "regression",
        "matching",
    ]  # Add 'dml', 'dowhy' if desired

    # ROI Params
    COST_PER_PROMO_INSTANCE = 0.10
    MARGIN_PERCENT = 0.30
    return (
        ANALYSES_TO_RUN,
        COST_PER_PROMO_INSTANCE,
        END_DATE,
        MARGIN_PERCENT,
        MATCHING_CALIPER,
        MATCHING_RATIO,
        NUM_PRODUCTS,
        NUM_STORES,
        RANDOM_SEED,
        START_DATE,
    )


@app.cell
def _(mo):
    mo.md(r"""### Helper Functions""")
    return


@app.cell
def _(mo):
    def run_and_summarize_analyses(
        analyzer, methods_to_run, matching_caliper, matching_ratio
    ):
        """Runs specified analyses, handles errors, returns results dict."""
        results = {}
        print("\n--- START: Running Analyses --- ")
        with mo.capture_stdout() as captured_stdout:
            for method in methods_to_run:
                print(f"  Calling {method}_analysis...")
                try:
                    if method == "naive":
                        results["naive"] = analyzer.naive_promotion_impact()
                    elif method == "regression":
                        results["regression"] = analyzer.regression_adjustment()
                    elif method == "matching":
                        results["matching"] = analyzer.matching_analysis(
                            caliper=matching_caliper, ratio=matching_ratio
                        )
                    elif method == "dml":  # Example for extending
                        # Optional: DML and DoWhy (potentially slow)
                        # print("  Calling double_ml_forest...")
                        # results['dml'] = analyzer.double_ml_forest()
                        print("    Skipping DML (commented out)...")
                        results["dml"] = {"skipped": True}
                    elif method == "dowhy":  # Example for extending
                        # print("  Calling dowhy_analysis...")
                        # results['dowhy'] = analyzer.dowhy_analysis()
                        print("    Skipping DoWhy (commented out)...")
                        results["dowhy"] = {"skipped": True}
                    else:
                        print(f"    Unknown analysis method: {method}")
                        results[method] = {"error": f"Unknown method {method}"}

                    if (
                        method in results
                        and "error" not in results[method]
                        and "skipped" not in results[method]
                    ):
                        print(f"  Finished {method}_analysis.")
                except Exception as e:
                    print(f"  ERROR running {method}_analysis: {e}")
                    results[method] = {"error": str(e)}
        print("--- END: Running Analyses --- ")
        mo.md("_(Analysis logs viewable in terminal where Marimo was run)_")
        # Optionally display captured stdout
        # mo.md(f"<details><summary>Analysis Logs</summary>```\n{captured_stdout.getvalue()}\n```</details>")
        return results

    return (run_and_summarize_analyses,)


@app.function
def format_num(val, fmt):
    """Helper to safely format numbers or return 'N/A'."""
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return "N/A"


@app.cell
def _(mo, np, pd):
    def run_and_display_counterfactuals(analyzer):
        """Defines, runs, and displays counterfactual scenarios."""
        mo.md("### Counterfactual Analysis Examples")
        print("\n--- START: Counterfactual Analysis --- ")
        cf_results = {}  # Store results keyed by scenario name

        try:
            # --- Define Scenarios ---
            scenarios = {}
            # Scenario 1: Always on promotion
            scenarios["Always On Promotion"] = {"promotion_applied": 1}

            # Scenarios 2 & 3 require analysis_data from the analyzer
            if (
                hasattr(analyzer, "analysis_data")
                and analyzer.analysis_data is not None
            ):
                cf_index = analyzer.analysis_data.index
                # Scenario 2: Random 30% promo
                scenarios["Random Subset (30% promo)"] = {
                    "promotion_applied": pd.Series(
                        np.random.random(len(cf_index)) < 0.3, index=cf_index
                    ).astype(int)
                }
                # Scenario 3: Prices increase 10%
                if "price" in analyzer.analysis_data.columns:
                    base_price_series = analyzer.analysis_data["price"]
                    scenarios["All Prices Increase 10%"] = {
                        "price": base_price_series * 1.1
                    }
                else:
                    print(
                        "--- WARNING: 'price' column not found in analyzer.analysis_data for Scenario 3 ---"
                    )
            else:
                print(
                    "--- WARNING: Cannot define scenarios accurately without analyzer.analysis_data ---"
                )

            # --- Run Scenarios ---
            with mo.capture_stdout() as captured_stdout_cf:
                for name, scenario_spec in scenarios.items():
                    print(f"  Running counterfactual scenario: {name}...")
                    try:
                        cf_results[name] = analyzer.perform_counterfactual_analysis(
                            scenario_spec
                        )
                        print(f"  Finished counterfactual scenario: {name}.")
                    except Exception as e:
                        print(f"  ERROR running counterfactual {name}: {e}")
                        cf_results[name] = {"error": str(e)}
            print(
                f"--- END: Counterfactual Analysis (stdout: {captured_stdout_cf.getvalue()}) ---"
            )

            # --- Display Results ---
            def display_cf(result):
                if result is None:
                    return "Analysis did not run or failed."
                if "error" in result:
                    return f"Error: {result['error']}"
                actual = result.get("actual_mean_sales")
                cf = result.get("counterfactual_mean_sales")
                change = result.get("percentage_change")
                return (
                    f"Predicted Mean Sales: {format_num(cf, '.2f')} "
                    f"(vs Actual: {format_num(actual, '.2f')}). "
                    f"Change: **{format_num(change, '.2f')}%**"
                )

            for name, result in cf_results.items():
                mo.md(f"**Scenario:** {name}\n- {display_cf(result)}")

        except Exception as cf_e:
            # Catch errors during scenario definition or the overall process
            mo.md(f"**Error during Counterfactual Analysis setup/execution:** {cf_e}")
            print(f"--- ERROR: Counterfactual Execution/Setup Exception: {cf_e} ---")
            # Store a general error if needed, though individual errors are preferred
            cf_results["General Error"] = {"error": str(cf_e)}

        return cf_results  # Return dictionary of results

    return (run_and_display_counterfactuals,)


@app.cell
def _(mo):
    mo.md(r"""### Causal Analysis Demonstration""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Generating Sample Data for Causal Analysis...""")
    return


@app.cell
def _():
    from utils.data_generation import generate_synthetic_retail_data

    return (generate_synthetic_retail_data,)


@app.cell
def _(
    END_DATE,
    NUM_PRODUCTS,
    NUM_STORES,
    RANDOM_SEED,
    START_DATE,
    generate_synthetic_retail_data,
    mo,
    pd,
):
    try:
        # Call the function from utils/data_generation.py
        sales_df, product_df, store_df = generate_synthetic_retail_data(
            start_date_str=START_DATE,
            end_date_str=END_DATE,
            num_stores=NUM_STORES,
            num_products=NUM_PRODUCTS,
            seed=RANDOM_SEED,
            # Add other parameters here to override defaults if necessary
        )

        # Display sample data head
        mo.md("Sample Data Head:")
        mo.ui.table(sales_df.head())

    except Exception as data_gen_e:
        mo.md(f"**Error generating sample data:** {data_gen_e}")
        # Define empty dataframes to prevent downstream errors in demo
        sales_df = pd.DataFrame()
        product_df = pd.DataFrame()
        store_df = pd.DataFrame()
    return product_df, sales_df, store_df


@app.cell
def _(mo):
    mo.md(r"""#### Initialize and Run Analyzer""")
    return


@app.cell
def _(PromotionCausalAnalyzer, mo, product_df, sales_df, store_df):
    analyzer = None  # Initialize to None
    if not sales_df.empty:
        mo.md("### Initializing Promotion Causal Analyzer...")
        print("\n--- START: Initializing Analyzer --- ")

        try:
            sales_df_renamed = sales_df.rename(
                columns={"sales_units": "sales", "on_promotion": "promotion_applied"}
            )
            # Convert boolean treatment to integer for compatibility if needed by analyzer
            if (
                "promotion_applied" in sales_df_renamed.columns
                and sales_df_renamed["promotion_applied"].dtype == "bool"
            ):
                sales_df_renamed["promotion_applied"] = sales_df_renamed[
                    "promotion_applied"
                ].astype(int)

            # Pass the RENAMED DFs
            analyzer = PromotionCausalAnalyzer(sales_df_renamed, product_df, store_df)
            mo.md("Analyzer initialized successfully.")
            print("--- END: Initializing Analyzer --- ")

        except KeyError as ke:
            mo.md(
                f"**Error initializing analyzer (Rename Failed):** Missing column {ke} in original sales_df."
            )
            print(
                f"--- ERROR: Initializing Analyzer Failed (Rename): Missing column {ke} ---"
            )
            analyzer = None
        except Exception as init_e:
            mo.md(f"**Error initializing analyzer:** {init_e}")
            print(f"--- ERROR: Initializing Analyzer Failed: {init_e} ---")
            analyzer = None  # Ensure analyzer is None if init fails
    return (analyzer,)


@app.cell
def _(mo):
    mo.md(r"""#### Perform Analyses (using helper function)""")
    return


@app.cell
def _(
    ANALYSES_TO_RUN,
    COST_PER_PROMO_INSTANCE,
    MARGIN_PERCENT,
    MATCHING_CALIPER,
    MATCHING_RATIO,
    analyzer,
    mo,
    pd,
    run_and_display_counterfactuals,
    run_and_summarize_analyses,
):
    results_dict = {}
    if analyzer:
        # Call the helper function to run analyses
        results_dict = run_and_summarize_analyses(
            analyzer, ANALYSES_TO_RUN, MATCHING_CALIPER, MATCHING_RATIO
        )

        # --- Display Results ---
        mo.md("### Analysis Results Summary")
        print("--- START: Displaying Results --- ")
        results_summary = []
        # Helper moved to top of cell
        # def format_num(val, fmt): ...

        # Iterate through the methods that were supposed to run
        for method in ANALYSES_TO_RUN:
            if method not in results_dict:
                results_summary.append(
                    {"Method": f"{method.title()} (Not Run)", "Comment": "-"}
                )
                continue  # Skip if not even present in results

            res = results_dict[method]
            if res.get("error"):
                results_summary.append(
                    {
                        "Method": method.title(),
                        "Comment": f"Error: {res['error'][:100]}...",
                    }
                )
            elif res.get("skipped"):
                results_summary.append({"Method": method.title(), "Comment": "Skipped"})
            else:
                # Format results based on method
                if method == "naive":
                    results_summary.append(
                        {
                            "Method": "Naive Difference",
                            "Estimated Effect (Abs Lift)": format_num(
                                res.get("absolute_lift_naive"), ".3f"
                            ),
                            "Estimated Effect (% Lift)": format_num(
                                res.get("percent_lift_naive"), ".2f"
                            )
                            + "%",
                            "Comment": "Correlation, likely biased",
                        }
                    )
                elif method == "regression":
                    confounders = res.get("confounders_used", [])
                    results_summary.append(
                        {
                            "Method": "Regression Adjustment",
                            "Estimated Effect (Abs Lift)": format_num(
                                res.get("estimated_ATE"), ".3f"
                            ),
                            "Estimated Effect (% Lift)": format_num(
                                res.get("percent_lift_pred"), ".2f"
                            )
                            + "%",
                            "Comment": f"Adjusts for {len(confounders)} observables"
                            + (
                                f": {', '.join(confounders[:3])}..."
                                if confounders
                                else ""
                            ),
                        }
                    )
                elif method == "matching":
                    results_summary.append(
                        {
                            "Method": "Propensity Score Matching",
                            "Estimated Effect (Abs Lift)": format_num(
                                res.get("estimated_ATE"), ".3f"
                            ),
                            "Estimated Effect (% Lift)": format_num(
                                res.get("percent_lift_est_ATE"), ".2f"
                            )
                            + "%",
                            "Comment": f"{res.get('matched_treatment_units', 0)} matched units "
                            f"(Caliper: {format_num(res.get('caliper', MATCHING_CALIPER), '.2f')}, "
                            f"Ratio: {res.get('ratio', MATCHING_RATIO)})",
                        }
                    )
                # Add formatting for 'dml', 'dowhy' if/when implemented
                # elif method == "dml": ...
                # elif method == "dowhy": ...
                else:
                    # Default for unknown but successful methods?
                    results_summary.append(
                        {
                            "Method": method.title(),
                            "Comment": "Completed (no summary format)",
                        }
                    )

        if results_summary:
            # Convert to DataFrame before displaying with mo.ui.table
            summary_df = pd.DataFrame(results_summary)
            try:
                mo.ui.table(summary_df)
            except Exception as table_e:
                mo.md(f"**Error displaying summary table:** {table_e}")
        else:
            mo.md("No analysis results could be generated or displayed.")
        print("--- END: Displaying Results --- ")

        # --- ROI Calculation ---
        mo.md("### ROI Calculation Example")
        print("\n--- START: Calculating ROI --- ")
        # Configuration for ROI (now moved to top)

        roi_result = None
        try:
            with mo.capture_stdout() as captured_stdout_roi:
                roi_result = analyzer.calculate_promotion_roi(
                    COST_PER_PROMO_INSTANCE,
                    MARGIN_PERCENT,  # Use config vars
                )
            print(
                f"--- END: Calculating ROI (stdout: {captured_stdout_roi.getvalue()}) ---"
            )

            if roi_result is None:
                mo.md("ROI calculation did not return a result.")
                print("--- WARNING: ROI Calculation returned None ---")
            elif "error" in roi_result:
                mo.md(f"**ROI Calculation Error:** {roi_result['error']}")
                print(f"--- ERROR: ROI Calculation Failed: {roi_result['error']} ---")
            elif "warning" in roi_result:
                mo.md(f"**ROI Calculation Warning:** {roi_result['warning']}")
                print(f"--- WARNING: ROI Calculation: {roi_result['warning']} ---")
            else:
                mo.md(
                    f"Assuming Promotion Cost = **${format_num(COST_PER_PROMO_INSTANCE, '.2f')}** per instance "
                    f"and Margin = **{format_num(MARGIN_PERCENT * 100, '.0f')}%**:"
                )
                mo.md(
                    f"- Estimated Total Incremental Margin: **${format_num(roi_result.get('total_incremental_margin_est'), '.2f')}**"
                )
                mo.md(
                    f"- Estimated Total Promotion Cost: **${format_num(roi_result.get('total_promotion_cost_est'), '.2f')}**"
                )
                roi_pct = roi_result.get("estimated_ROI_percent")
                mo.md(f"- Estimated ROI: **{format_num(roi_pct, '.2f')}%**")
                # Check if profitable_estimate key exists and is True
                profitable = roi_result.get("profitable_estimate")
                if profitable is not None:
                    mo.md(
                        f"  - Profitable Estimate: **{'Yes' if profitable else 'No'}**"
                    )
        except Exception as roi_e:
            mo.md(f"**Error during ROI Calculation execution:** {roi_e}")
            print(f"--- ERROR: ROI Execution Exception: {roi_e} ---")
            roi_result = {"error": str(roi_e)}  # Store error

        # --- Counterfactual Analysis ---
        # Now call the helper function

        cf_results_dict = run_and_display_counterfactuals(analyzer)

        # Assign individual results if needed downstream (optional)
        cf_result1 = cf_results_dict.get("Always On Promotion")
        cf_result2 = cf_results_dict.get("Random Subset (30% promo)")
        cf_result3 = cf_results_dict.get("All Prices Increase 10%")

    else:  # Analyzer was not initialized
        mo.md(
            "Analyzer could not be initialized. Skipping analyses, ROI, and counterfactuals."
        )
        # Ensure results variables exist but are None
        results_dict = {}
        roi_result = None
        cf_result1, cf_result2, cf_result3 = None, None, None

    # Returning relevant results. Adjust as needed for notebook flow.
    # If these are needed by later cells, return them.
    # Otherwise, return None or an empty tuple.
    return (cf_result2,)


@app.cell
def __cell7():
    return


@app.cell
def _(cf_result2, mo):
    # Check if cf_result2 has a value before trying to display it
    if cf_result2 is not None:
        cf_result2
    else:
        # Display a message if the result is None (analysis likely failed)
        mo.md("Counterfactual analysis (Scenario 2) did not complete successfully.")
    return


@app.cell
def _(cf_result2):
    cf_result2
    return


if __name__ == "__main__":
    app.run()
