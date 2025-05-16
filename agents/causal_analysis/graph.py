import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def define_causal_graph(
    analysis_data: pd.DataFrame,
    treatment: str = "promotion_applied",
    outcome: str = "sales",
    common_causes: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> tuple[str, str, str, list[str]]:
    """
    Defines the causal graph structure as a string and identifies common causes.

    Args:
        analysis_data: The dataframe containing the analysis variables.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        common_causes: List of variable names known to be common causes.
                       If None, infers potential numeric common causes from data columns.
        exclude_cols: List of column names to exclude from inferred common causes.
                      Defaults to ['date'] if None.

    Returns:
        A tuple containing:
        - str: The causal graph in DOT format.
        - str: The name of the treatment variable used.
        - str: The name of the outcome variable used.
        - list[str]: The list of common cause variable names used.
    """
    if exclude_cols is None:
        exclude_cols = ["date"]  # Default columns to exclude from inference

    if common_causes is None:
        # Infer potential common causes from columns, excluding treatment, outcome, and specified cols
        potential_causes = [
            col
            for col in analysis_data.columns
            if col not in [treatment, outcome] + exclude_cols
            and pd.api.types.is_numeric_dtype(analysis_data[col])  # Basic check for numeric features
        ]
        # Heuristic: Use all numeric ones found, could be refined with domain knowledge
        common_causes = potential_causes
        print(f"Inferred common causes: {common_causes}")
    else:
        # Validate provided common causes exist in the data
        missing_causes = [c for c in common_causes if c not in analysis_data.columns]
        if missing_causes:
            raise ValueError(f"Provided common causes not found in data: {missing_causes}")

    # Basic graph: Common causes affect both treatment and outcome
    graph = "digraph {\n"
    graph += f'  "{treatment}" [label="{treatment.replace("_", " ").title()}"];\n'  # Use actual name
    graph += f'  "{outcome}" [label="{outcome.replace("_", " ").title()}"];\n'  # Use actual name

    # Add nodes for common causes
    for cause in common_causes:
        graph += f'  "{cause}" [label="{cause.replace("_", " ").title()}"];\n'

    # Add edges from common causes to treatment and outcome
    for cause in common_causes:
        graph += f'  "{cause}" -> "{treatment}";\n'
        graph += f'  "{cause}" -> "{outcome}";\n'

    # Add edge from treatment to outcome (the effect we want to estimate)
    graph += f'  "{treatment}" -> "{outcome}";\n'

    graph += "}"

    return graph, treatment, outcome, common_causes


def visualize_causal_graph(graph_str: str, save_path: str | None = None):
    """
    Visualizes the causal graph defined in DOT format using NetworkX and Matplotlib.

    Requires graphviz to be installed (`pip install graphviz` and potentially system install).

    Args:
        graph_str: The causal graph in DOT format string.
        save_path: Optional path to save the visualization. If None, displays the plot.
    """
    try:
        # Use NetworkX to parse the DOT string and draw
        G = nx.drawing.nx_pydot.read_dot(graph_str)  # Requires pydot

        plt.figure(figsize=(10, 6))
        # Use a layout that works well for directed graphs
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")  # Requires graphviz

        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=nx.get_node_attributes(G, "label"),  # Use labels from DOT
            node_size=3000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            arrows=True,
            arrowstyle="-|>",
        )

        plt.title("Causal Graph")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Causal graph saved to {save_path}")
            plt.close()  # Close the plot window if saving
        else:
            plt.show()

    except ImportError as e:
        print(f"Error visualizing graph: {e}. Make sure pydot and graphviz are installed.")
    except FileNotFoundError:
        # This error occurs if the 'dot' executable from graphviz is not found
        print(
            "Error visualizing graph: 'dot' command not found. "
            "Please ensure Graphviz is installed and in your system's PATH."
            " You can usually install it via your system package manager (e.g., `brew install graphviz` on macOS, `sudo apt-get install graphviz` on Debian/Ubuntu)."
        )

    except Exception as e:
        print(f"An unexpected error occurred during graph visualization: {e}")
        import traceback

        traceback.print_exc()
