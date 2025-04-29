import marimo

__generated_with = "0.13.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Chapter 6: Foundation Models and Visual Intelligence

        This chapter explores how Foundation Models, powered by large language models and advanced visual intelligence, redefine responsiveness and adaptability in retail environments. 

        You'll discover how integrating these powerful AI capabilities can enable real-time shelf monitoring, improved customer interactions, and intelligent product recognition. Additionally, the chapter dives into Knowledge Graphs and Semantic Reasoning, illustrating how structured knowledge and ontologies significantly enhance decision accuracy, personalization, and overall retail intelligence. 

        By combining these critical technologies, you'll be equipped to build sophisticated AI-driven retail experiences that seamlessly blend perception, language, and reasoning.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Imports and logger setup""")
    return


@app.cell
def _():
    import os
    import asyncio
    import pandas as pd
    import marimo as mo


    # Setup basic logging
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Import DummyDB and DummyOrderSystem from connectors
    from connectors.dummy_db import DummyDB
    from connectors.dummy_order_system import DummyOrderSystem

    # Import DummyPlanogramDB and DummyInventorySystem from connectors
    from connectors.dummy_planogram_db import DummyPlanogramDB
    from connectors.dummy_inventory_system import DummyInventorySystem

    # Import ShelfMonitoringAgent
    from agents.cv import ShelfMonitoringAgent

    return (
        DummyDB,
        DummyInventorySystem,
        DummyOrderSystem,
        DummyPlanogramDB,
        ShelfMonitoringAgent,
        asyncio,
        logger,
        mo,
        os,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## LLM-Powered Customer Service Agent

        The following example demonstrates how an LLM can be integrated into a retail customer service system, combining natural language understanding with structured business logic
        """
    )
    return


@app.cell
def _():
    from agents.llm import RetailCustomerServiceAgent

    return (RetailCustomerServiceAgent,)


@app.cell
def _(DummyDB, DummyOrderSystem, RetailCustomerServiceAgent, os):
    # Define dummy database/system connectors for demonstration
    policy_guidelines = {
        "returns": {"return_window_days": 30, "methods": ["Mail", "Store"]}
    }
    api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

    customer_service_agent = RetailCustomerServiceAgent(
        DummyDB(), DummyOrderSystem(), DummyDB(), policy_guidelines, api_key
    )
    return (customer_service_agent,)


@app.cell
def _(customer_service_agent, mo):
    # Cell 7: Run LLM Agent Demo
    async def run_customer_interaction_demo():
        customer_id = "C123"  # Alice
        messages = [
            "Hi, what's the status of my latest order?",  # ORD988
            "Can I return the yoga mat from order ORD987?",  # Eligible
            "I want to return my running shoes from order ORD988.",  # Not delivered
            "Do you have that yoga mat in blue?",  # Identifier exists, details retrieved
        ]
        responses_md = []

        if (
            not hasattr(customer_service_agent, "client")
            or customer_service_agent.client is None
        ):
            return mo.md(
                "**Error:** OpenAI client not initialized. Cannot run LLM demo. Please set `OPENAI_API_KEY`."
            )

        responses_md.append(mo.md("### Customer Service Agent Demo"))
        for msg in messages:
            response = await customer_service_agent.process_customer_inquiry(
                customer_id, msg
            )
            responses_md.append(
                mo.md(
                    f"**Customer:** {msg}\n\n**Agent:** {response.get('message', 'Error')}\n_(Actions: {response.get('actions', [])})_"
                )
            )
            responses_md.append("---")

        return mo.vstack(responses_md)

    return (run_customer_interaction_demo,)


@app.cell
async def _(logger, mo, run_customer_interaction_demo):
    # Run demo safely in Marimo/Jupyter environment
    interaction_output = None
    try:
        # Simply await the async function directly
        # Marimo will run this within its existing event loop
        interaction_output = await run_customer_interaction_demo()
    except Exception as e:
        logger.error(f"Error running customer interaction demo: {e}", exc_info=True)
        interaction_output = mo.md(f"**Error running LLM Demo:** {e}")

    # Display the output in this cell
    interaction_output
    return


@app.cell
def _(mo):
    mo.md(r"""## Shelf Monitoring Agent""")
    return


@app.cell
def _(DummyInventorySystem, DummyPlanogramDB, ShelfMonitoringAgent):
    # Instantiation
    model_path = ""  # <<<--- UPDATE
    cam_urls = {"CAM01": "0", "CAM02": "1"}

    shelf_agent = ShelfMonitoringAgent(
        model_path,
        DummyPlanogramDB(),
        DummyInventorySystem(),
        cam_urls,
        confidence_threshold=0.5,
        check_frequency_seconds=15,
    )

    return (shelf_agent,)


@app.cell
def _(asyncio, logger, mo, pd, shelf_agent):
    # Cell 10: Define Shelf Monitoring Demo Function
    async def run_shelf_monitoring_demo():
        if not shelf_agent or not shelf_agent.detection_model:
            # Use mo.stop() to prevent further execution if setup fails
            mo.stop(
                True,
                mo.md(
                    "**Error:** Shelf Agent/Model not loaded. Check `model_path` & TF install."
                ),
            )
            # The return below is now technically unreachable due to mo.stop,
            # but kept for clarity if mo.stop is removed later.
            return mo.md(
                "**Error:** Shelf Agent/Model not loaded. Check `model_path` & TF install."
            )

        loc = "LOC1"
        sections = ["SEC001"]  # Only monitor SEC001 (CAM01='0') by default
        cam1_src = shelf_agent.camera_streams.get("CAM01", "N/A")
        output = [
            mo.md("## Shelf Monitoring Demo"),
            mo.md(f"Monitoring sections: {sections} at {loc}..."),
            mo.md(
                f"_Ensure CAM01 (src: '{cam1_src}') active & sees target items (e.g., cans/bottles). Check console logs._"
            ),
        ]

        try:
            start_tasks = [
                shelf_agent.start_monitoring_section(loc, s) for s in sections
            ]
            await asyncio.gather(*start_tasks)

            duration = 35  # seconds
            output.append(mo.md(f"Monitoring for **{duration} seconds**..."))
            await asyncio.sleep(duration)
            output.append(mo.md("Stopping monitoring..."))
            await shelf_agent.stop_all_monitoring()
            output.append(mo.md("Monitoring stopped."))
            output.append(mo.md("---"))

            for section in sections:
                final = shelf_agent.detected_issues.get(section, [])
                output.append(mo.md(f"**Final Issues for {section} ({len(final)}):**"))
                if final:
                    try:
                        output.append(mo.ui.table(pd.DataFrame(final), selection=None))
                    except Exception as e:
                        output.append(mo.md(f"_Table error: {e}_ ```\n{final}\n```"))
                else:
                    output.append(mo.md("- No issues detected."))
                output.append("---")
        except Exception as e:
            logger.error(f"Error during shelf monitoring execution: {e}", exc_info=True)
            output.append(mo.md(f"**Runtime Error during Shelf Demo:** {e}"))
            # Ensure monitoring stops even if there's an error during the run
            await shelf_agent.stop_all_monitoring()
            output.append(mo.md("Monitoring stopped due to error."))

        return mo.vstack(output)

    # Return the function itself so it can be called in the next cell
    return (run_shelf_monitoring_demo,)


@app.cell
async def _(logger, mo, run_shelf_monitoring_demo):
    # Cell 11: Run Shelf Monitoring Demo
    monitoring_output = None
    try:
        # Marimo will automatically await this top-level async call
        monitoring_output = await run_shelf_monitoring_demo()
    except Exception as e:
        # Catch errors from the function call itself (e.g., if the function wasn't returned properly)
        logger.error(f"Error calling run_shelf_monitoring_demo: {e}", exc_info=True)
        monitoring_output = mo.md(f"**Error setting up Shelf Demo:** {e}")

    # Display the output
    monitoring_output
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Integration with Other Agent Systems

        Computer vision systems are most valuable when integrated with other retail agent capabilities:

        1. **Computer Vision + LLMs**: Enable natural language queries about visual store conditions, such as "Show me all sections with more than 20% out-of-stocks" or "Which endcaps need to be reset for the new promotion?"
        2. **Computer Vision + IoT**: Correlate visual data with shelf weight sensors to distinguish between similar-looking products or verify that observed changes match weight changes.
        3. **Computer Vision + Knowledge Graphs**: Enrich product recognition with semantic relationships, allowing agents to understand not just what they see but what it means in the retail context.
        4. **Computer Vision + Robotic Systems**: Direct autonomous robots to respond to detected issues, such as cleaning spills, retrieving products, or scanning barcodes to verify inventory.
        """
    )
    return


@app.cell
def _():
    # Cell 12: Final app definition cell
    # Define the main execution block for the Marimo app.
    if __name__ == "__main__":
        pass  # Marimo handles execution
    return


if __name__ == "__main__":
    app.run()
