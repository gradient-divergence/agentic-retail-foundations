import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def __cell1():
    # Set matplotlib to use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    import logging
    import pathlib

    import marimo as mo
    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)
    # Basic config for the logger to ensure it prints to stdout for capture if needed
    # This might be duplicative if Marimo or another part of the setup already configures root logger
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    return mo, np, pathlib, pd, logging, logger


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
    knowledge_graph_path = pathlib.Path("assets/knowledge-graph-for-retail-product-relationships.svg")

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
    causal_inference_path = pathlib.Path("assets/causal-inference-in-promotion-effectiveness.svg")
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
def _(mo, logger):
    def run_and_summarize_analyses(analyzer, methods_to_run, matching_caliper, matching_ratio):
        """Runs specified analyses, handles errors, returns results dict."""
        results = {}
        logger.info("\n--- START: Running Analyses --- ")
        with mo.capture_stdout():
            for method in methods_to_run:
                logger.info(f"  Calling {method}_analysis...")
                try:
                    if method == "naive":
                        results["naive"] = analyzer.naive_promotion_impact()
                    elif method == "regression":
                        results["regression"] = analyzer.regression_adjustment()
                    elif method == "matching":
                        results["matching"] = analyzer.matching_analysis(caliper=matching_caliper, ratio=matching_ratio)
                    elif method == "dml":
                        logger.info("    Skipping DML (commented out)...")
                        results["dml"] = {"skipped": True}
                    elif method == "dowhy":
                        logger.info("    Skipping DoWhy (commented out)...")
                        results["dowhy"] = {"skipped": True}
                    else:
                        logger.warning(f"    Unknown analysis method: {method}")
                        results[method] = {"error": f"Unknown method {method}"}

                    if method in results and "error" not in results[method] and "skipped" not in results[method]:
                        logger.info(f"  Finished {method}_analysis.")
                except Exception as e:
                    logger.error(f"  ERROR running {method}_analysis: {e}")
                    results[method] = {"error": str(e)}
        logger.info("--- END: Running Analyses --- ")
        mo.md("_(Analysis logs viewable in terminal where Marimo was run)_")
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
def _(mo, np, pd, logger):
    def _define_counterfactual_scenarios(analyzer_inst, pd_module, np_module, logger_inst):
        scenarios = {}
        scenarios["Always On Promotion"] = {"promotion_applied": 1}
        if hasattr(analyzer_inst, "analysis_data") and analyzer_inst.analysis_data is not None:
            cf_index = analyzer_inst.analysis_data.index
            scenarios["Random Subset (30% promo)"] = {
                "promotion_applied": pd_module.Series(np_module.random.random(len(cf_index)) < 0.3, index=cf_index).astype(int)
            }
            if "price" in analyzer_inst.analysis_data.columns:
                base_price_series = analyzer_inst.analysis_data["price"]
                scenarios["All Prices Increase 10%"] = {"price": base_price_series * 1.1}
            else:
                logger_inst.warning("--- WARNING: 'price' column not found for Scenario 3 ---")
        else:
            logger_inst.warning("--- WARNING: Cannot define scenarios without analyzer.analysis_data ---")
        return scenarios

    return (_define_counterfactual_scenarios,)


@app.cell
def _(logger, mo):
    def _run_counterfactual_scenarios(analyzer_inst, scenarios_map, logger_inst, marimo_mo):
        cf_results_internal = {}
        with marimo_mo.capture_stdout() as captured_stdout_cf_run:
            for name, scenario_spec in scenarios_map.items():
                logger_inst.info(f"  Running counterfactual scenario: {name}...")
                try:
                    cf_results_internal[name] = analyzer_inst.perform_counterfactual_analysis(scenario_spec)
                    logger_inst.info(f"  Finished counterfactual scenario: {name}.")
                except Exception as e:
                    logger_inst.error(f"  ERROR running counterfactual {name}: {e}")
                    cf_results_internal[name] = {"error": str(e)}
        logger_inst.info(f"--- END: Counterfactual Analysis (stdout: {captured_stdout_cf_run.getvalue()}) ---")
        return cf_results_internal

    return (_run_counterfactual_scenarios,)


@app.cell
def _(mo, format_num):
    def _display_counterfactual_results(cf_results_to_display, marimo_mo, format_num_func):
        for name, result in cf_results_to_display.items():
            if result is None:
                display_text = "Analysis did not run or failed."
            elif "error" in result:
                display_text = f"Error: {result['error']}"
            else:
                actual = result.get("actual_mean_sales")
                cf_val = result.get("counterfactual_mean_sales")
                change = result.get("percentage_change")
                display_text = (
                    f"Predicted Mean Sales: {format_num_func(cf_val, '.2f')} "
                    f"(vs Actual: {format_num_func(actual, '.2f')}). "
                    f"Change: **{format_num_func(change, '.2f')}%**"
                )
            marimo_mo.md(f"**Scenario:** {name}\\n- {display_text}")

    return (_display_counterfactual_results,)


@app.cell
def _(mo, logger, pd, np, format_num, _define_counterfactual_scenarios, _run_counterfactual_scenarios, _display_counterfactual_results):
    def run_and_display_counterfactuals(analyzer_inst):
        """Defines, runs, and displays counterfactual scenarios using helper functions."""
        mo.md("### Counterfactual Analysis Examples")
        logger.info("\n--- START: Main Counterfactual Analysis Orchestrator ---")
        cf_results = {}
        try:
            scenarios = _define_counterfactual_scenarios(analyzer_inst, pd, np, logger)
            cf_results = _run_counterfactual_scenarios(analyzer_inst, scenarios, logger, mo)
            _display_counterfactual_results(cf_results, mo, format_num)
        except Exception as cf_e:
            mo.md(f"**Error during Counterfactual Analysis setup/execution:** {cf_e}")
            logger.error(f"--- ERROR: Counterfactual Execution/Setup Exception: {cf_e} ---")
            cf_results["General Error"] = {"error": str(cf_e)}
        logger.info("--- END: Main Counterfactual Analysis Orchestrator ---")
        return cf_results

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
def _(PromotionCausalAnalyzer, mo, product_df, sales_df, store_df, logger):
    analyzer = None  # Initialize to None
    if not sales_df.empty:
        mo.md("### Initializing Promotion Causal Analyzer...")
        logger.info("\n--- START: Initializing Analyzer --- ")

        try:
            sales_df_renamed = sales_df.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
            # Convert boolean treatment to integer for compatibility if needed by analyzer
            if "promotion_applied" in sales_df_renamed.columns and sales_df_renamed["promotion_applied"].dtype == "bool":
                sales_df_renamed["promotion_applied"] = sales_df_renamed["promotion_applied"].astype(int)

            # Pass the RENAMED DFs
            analyzer = PromotionCausalAnalyzer(sales_df_renamed, product_df, store_df)
            mo.md("Analyzer initialized successfully.")
            logger.info("--- END: Initializing Analyzer --- ")

        except KeyError as ke:
            mo.md(f"**Error initializing analyzer (Rename Failed):** Missing column {ke} in original sales_df.")
            logger.error(f"--- ERROR: Initializing Analyzer Failed (Rename): Missing column {ke} ---")
            analyzer = None
        except Exception as init_e:
            mo.md(f"**Error initializing analyzer:** {init_e}")
            logger.error(f"--- ERROR: Initializing Analyzer Failed: {init_e} ---")
            analyzer = None  # Ensure analyzer is None if init fails
    return (analyzer,)


@app.cell
def _(mo):
    mo.md(r"""#### Perform Analyses (using helper function)""")
    return


@app.cell
def _(  # Cell 1: Analysis Execution
    ANALYSES_TO_RUN,
    MATCHING_CALIPER,
    MATCHING_RATIO,
    analyzer,
    run_and_summarize_analyses,
    logger,
    mo,  # Added mo for conditional message
):
    results_dict = {}
    if analyzer:
        logger.info("--- Cell: Executing Causal Analyses ---")
        results_dict = run_and_summarize_analyses(analyzer, ANALYSES_TO_RUN, MATCHING_CALIPER, MATCHING_RATIO)
        logger.info("--- Cell: Causal Analyses Execution Complete ---")
    else:
        logger.warning("Analyzer not initialized, skipping analyses execution cell.")
        mo.md("Analyzer not initialized. Skipping Causal Analyses.")
    return results_dict


@app.cell
def _(  # Cell 2: Display Analysis Results
    ANALYSES_TO_RUN,
    MATCHING_CALIPER,
    MATCHING_RATIO,
    analyzer,
    mo,
    pd,
    results_dict,
    format_num,
    logger,
):
    def _format_naive_result(res, format_num_func):
        return {
            "Method": "Naive Difference",
            "Estimated Effect (Abs Lift)": format_num_func(res.get("absolute_lift_naive"), ".3f"),
            "Estimated Effect (% Lift)": format_num_func(res.get("percent_lift_naive"), ".2f") + "%",
            "Comment": "Correlation, likely biased",
        }

    def _format_regression_result(res, format_num_func):
        confounders = res.get("confounders_used", [])
        return {
            "Method": "Regression Adjustment",
            "Estimated Effect (Abs Lift)": format_num_func(res.get("estimated_ATE"), ".3f"),
            "Estimated Effect (% Lift)": format_num_func(res.get("percent_lift_pred"), ".2f") + "%",
            "Comment": f"Adjusts for {len(confounders)} observables" + (f": {', '.join(confounders[:3])}..." if confounders else ""),
        }

    def _format_matching_result(res, format_num_func, caliper_val, ratio_val):
        return {
            "Method": "Propensity Score Matching",
            "Estimated Effect (Abs Lift)": format_num_func(res.get("estimated_ATE"), ".3f"),
            "Estimated Effect (% Lift)": format_num_func(res.get("percent_lift_est_ATE"), ".2f") + "%",
            "Comment": f"{res.get('matched_treatment_units', 0)} matched units "
            f"(Caliper: {format_num_func(res.get('caliper', caliper_val), '.2f')}, "
            f"Ratio: {format_num_func(res.get('ratio', ratio_val))})",
        }

    if analyzer and results_dict:
        logger.info("--- Cell: Displaying Analysis Results ---")
        mo.md("### Analysis Results Summary")
        results_summary = []
        for method in ANALYSES_TO_RUN:
            if method not in results_dict:
                results_summary.append({"Method": f"{method.title()} (Not Run)", "Comment": "-"})
                continue
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
                if method == "naive":
                    results_summary.append(_format_naive_result(res, format_num))
                elif method == "regression":
                    results_summary.append(_format_regression_result(res, format_num))
                elif method == "matching":
                    results_summary.append(_format_matching_result(res, format_num, MATCHING_CALIPER, MATCHING_RATIO))
                else:
                    results_summary.append(
                        {
                            "Method": method.title(),
                            "Comment": "Completed (no summary format)",
                        }
                    )
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            try:
                mo.ui.table(summary_df)
            except Exception as table_e:
                mo.md(f"**Error displaying summary table:** {table_e}")
        else:
            mo.md("No analysis results could be generated or displayed.")
        logger.info("--- Cell: Analysis Results Display Complete ---")
    elif analyzer:
        mo.md("Analysis results dictionary not available for display.")
    return


@app.cell
def _(COST_PER_PROMO_INSTANCE, MARGIN_PERCENT, analyzer, mo, logger, format_num):  # Cell 3: ROI Calculation
    roi_result = None
    if analyzer:
        logger.info("--- Cell: Calculating ROI ---")
        mo.md("### ROI Calculation Example")
        try:
            with mo.capture_stdout() as captured_stdout_roi_cell:  # Renamed to avoid conflict
                roi_result = analyzer.calculate_promotion_roi(
                    COST_PER_PROMO_INSTANCE,
                    MARGIN_PERCENT,
                )
            logger.info(f"--- END: Calculating ROI (stdout: {captured_stdout_roi_cell.getvalue()}) ---")

            if roi_result is None:
                mo.md("ROI calculation did not return a result.")
                logger.warning("--- WARNING: ROI Calculation returned None ---")
            elif "error" in roi_result:
                mo.md(f"**ROI Calculation Error:** {roi_result['error']}")
                logger.error(f"--- ERROR: ROI Calculation Failed: {roi_result['error']} ---")
            elif "warning" in roi_result:
                mo.md(f"**ROI Calculation Warning:** {roi_result['warning']}")
                logger.warning(f"--- WARNING: ROI Calculation: {roi_result['warning']} ---")
            else:
                mo.md(
                    f"Assuming Promotion Cost = **${format_num(COST_PER_PROMO_INSTANCE, '.2f')}** per instance "
                    f"and Margin = **{format_num(MARGIN_PERCENT * 100, '.0f')}%**:"
                )
                mo.md(f"- Estimated Total Incremental Margin: **${format_num(roi_result.get('total_incremental_margin_est'), '.2f')}**")
                mo.md(f"- Estimated Total Promotion Cost: **${format_num(roi_result.get('total_promotion_cost_est'), '.2f')}**")
                roi_pct = roi_result.get("estimated_ROI_percent")
                mo.md(f"- Estimated ROI: **{format_num(roi_pct, '.2f')}%**")
                profitable = roi_result.get("profitable_estimate")
                if profitable is not None:
                    mo.md(f"  - Profitable Estimate: **{'Yes' if profitable else 'No'}**")
        except Exception as roi_e:
            mo.md(f"**Error during ROI Calculation execution:** {roi_e}")
            logger.error(f"--- ERROR: ROI Execution Exception: {roi_e} ---")
            roi_result = {"error": str(roi_e)}
        logger.info("--- Cell: ROI Calculation Complete ---")
    else:
        logger.warning("Analyzer not initialized, skipping ROI calculation cell.")
        mo.md("Analyzer not initialized. Skipping ROI Calculation.")
    return roi_result


@app.cell
def _(analyzer, run_and_display_counterfactuals, logger, mo):  # Cell 4: Counterfactual Execution
    cf_results_dict = None
    cf_result_display_placeholder = None

    if analyzer:
        logger.info("--- Cell: Running Counterfactual Analysis ---")
        cf_results_dict = run_and_display_counterfactuals(analyzer)
        logger.info("--- Cell: Counterfactual Analysis Complete ---")

        if cf_results_dict:
            if "Always On Promotion" in cf_results_dict:
                cf_result_display_placeholder = cf_results_dict["Always On Promotion"]
            elif "Random Subset (30% promo)" in cf_results_dict:
                cf_result_display_placeholder = cf_results_dict["Random Subset (30% promo)"]
            elif "All Prices Increase 10%" in cf_results_dict:
                cf_result_display_placeholder = cf_results_dict["All Prices Increase 10%"]
    else:
        logger.warning("Analyzer not initialized, skipping counterfactuals cell.")
        mo.md("Analyzer not initialized. Skipping Counterfactual Analysis.")

    return cf_results_dict, cf_result_display_placeholder


@app.cell
def __cell7():
    return


@app.cell
def _(cf_result_display_placeholder, mo):
    if cf_result_display_placeholder is not None:
        pass
    else:
        mo.md("Specific counterfactual scenario result not available for display.")
    return


@app.cell
def _(cf_result_display_placeholder):
    return


if __name__ == "__main__":
    app.run()
