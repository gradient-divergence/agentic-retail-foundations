# tests/agents/causal_analysis/test_graph.py

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions to test
from agents.causal_analysis.graph import define_causal_graph, visualize_causal_graph

@pytest.fixture
def sample_analysis_data() -> pd.DataFrame:
    """Fixture for sample data suitable for graph definition."""
    return pd.DataFrame({
        'sales': [10, 12, 5, 8, 11, 6],
        'promotion_applied': [1, 0, 1, 0, 1, 0],
        'price': [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        'marketing_spend': [100, 150, 100, 150, 120, 110],
        'day_of_week': [0, 0, 1, 1, 2, 2], # Example numeric feature
        'store_location': ['Urban', 'Suburban', 'Urban', 'Suburban', 'Urban', 'Suburban'] # Non-numeric
    })

# --- Tests for define_causal_graph ---

def test_define_graph_infer_causes(sample_analysis_data):
    """Test graph definition inferring common causes from numeric columns."""
    graph_str, treatment, outcome, common_causes = define_causal_graph(
        analysis_data=sample_analysis_data,
        treatment='promotion_applied',
        outcome='sales'
        # exclude_cols defaults to ['date'], which isn't present here anyway
    )

    expected_common_causes = sorted(['price', 'marketing_spend', 'day_of_week']) # All numeric except treatment/outcome
    assert treatment == 'promotion_applied'
    assert outcome == 'sales'
    assert sorted(common_causes) == expected_common_causes

    # Check graph structure (basic checks)
    assert 'digraph {' in graph_str
    assert '"promotion_applied"' in graph_str
    assert '"sales"' in graph_str
    for cause in expected_common_causes:
        assert f'"{cause}"' in graph_str
        assert f'"{cause}" -> "promotion_applied"' in graph_str
        assert f'"{cause}" -> "sales"' in graph_str
    assert '"promotion_applied" -> "sales"' in graph_str
    assert 'store_location' not in graph_str # Should exclude non-numeric

def test_define_graph_explicit_causes(sample_analysis_data):
    """Test graph definition with explicitly provided common causes."""
    explicit_causes = ['price', 'day_of_week']
    graph_str, _, _, common_causes = define_causal_graph(
        analysis_data=sample_analysis_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=explicit_causes
    )

    assert sorted(common_causes) == sorted(explicit_causes)
    assert 'marketing_spend' not in graph_str # Not included in explicit list
    assert '"price"' in graph_str
    assert '"day_of_week"' in graph_str
    assert '"price" -> "sales"' in graph_str
    assert '"day_of_week" -> "promotion_applied"' in graph_str

def test_define_graph_exclude_cols(sample_analysis_data):
    """Test excluding columns from inferred common causes."""
    _, _, _, common_causes = define_causal_graph(
        analysis_data=sample_analysis_data,
        treatment='promotion_applied',
        outcome='sales',
        exclude_cols=['price'] # Exclude price from inference
    )
    expected_common_causes = sorted(['marketing_spend', 'day_of_week'])
    assert sorted(common_causes) == expected_common_causes

def test_define_graph_missing_explicit_cause(sample_analysis_data):
    """Test ValueError when an explicit common cause is missing from data."""
    with pytest.raises(ValueError, match="Provided common causes not found in data:.*'missing_feature'"):
        define_causal_graph(
            analysis_data=sample_analysis_data,
            treatment='promotion_applied',
            outcome='sales',
            common_causes=['price', 'missing_feature']
        )

# --- Tests for visualize_causal_graph ---

@pytest.fixture
def sample_dot_graph() -> str:
    """A simple DOT graph string for testing visualization."""
    return "digraph { A -> B; B -> C; }"

@patch('matplotlib.pyplot.show')
@patch('networkx.drawing.nx_pydot.read_dot')
@patch('networkx.drawing.nx_pydot.graphviz_layout')
def test_visualize_graph_show(mock_layout, mock_read_dot, mock_show, sample_dot_graph):
    """Test that visualize_causal_graph calls plt.show() when no save_path is given."""
    # Mock the graphviz/pydot related calls to avoid dependency issues
    mock_graph = MagicMock()
    mock_read_dot.return_value = mock_graph
    mock_layout.return_value = {'A': (0,0), 'B': (1,1), 'C': (2,0)} # Dummy positions

    visualize_causal_graph(sample_dot_graph)

    mock_read_dot.assert_called_once_with(sample_dot_graph)
    mock_layout.assert_called_once()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('networkx.drawing.nx_pydot.read_dot')
@patch('networkx.drawing.nx_pydot.graphviz_layout')
def test_visualize_graph_save(mock_layout, mock_read_dot, mock_close, mock_savefig, sample_dot_graph):
    """Test that visualize_causal_graph calls plt.savefig() when save_path is given."""
    mock_graph = MagicMock()
    mock_read_dot.return_value = mock_graph
    mock_layout.return_value = {'A': (0,0), 'B': (1,1), 'C': (2,0)}
    save_path = "test_graph.png"

    visualize_causal_graph(sample_dot_graph, save_path=save_path)

    mock_read_dot.assert_called_once_with(sample_dot_graph)
    mock_layout.assert_called_once()
    mock_savefig.assert_called_once_with(save_path, bbox_inches="tight")
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.show')
@patch('networkx.drawing.nx_pydot.read_dot', side_effect=ImportError("pydot not found"))
def test_visualize_graph_import_error(mock_read_dot, mock_show, sample_dot_graph, capsys):
    """Test handling of ImportError if pydot/graphviz are missing."""
    visualize_causal_graph(sample_dot_graph)
    captured = capsys.readouterr()
    # Check stdout for the printed error message
    assert "Error visualizing graph" in captured.out
    assert "pydot not found" in captured.out
    assert captured.err == '' # Ensure nothing was printed to stderr
    mock_show.assert_not_called()

@patch('matplotlib.pyplot.show')
@patch('networkx.drawing.nx_pydot.read_dot')
@patch('networkx.drawing.nx_pydot.graphviz_layout', side_effect=FileNotFoundError("[Errno 2] No such file or directory: 'dot'..."))
def test_visualize_graph_graphviz_not_found(mock_layout, mock_read_dot, mock_show, sample_dot_graph, capsys):
    """Test handling of FileNotFoundError if graphviz 'dot' executable is missing."""
    mock_graph = MagicMock()
    mock_read_dot.return_value = mock_graph

    visualize_causal_graph(sample_dot_graph)

    captured = capsys.readouterr()
    # Check stdout for the printed error message
    assert "Error visualizing graph: 'dot' command not found" in captured.out
    assert captured.err == '' # Ensure nothing was printed to stderr
    mock_show.assert_not_called() 