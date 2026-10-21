"""Check the printed mechanisms, rather than just hashes or include existence."""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

BOOK_ROOT = Path(__file__).resolve().parents[4]
EXTRACTOR = BOOK_ROOT / "scripts/snippets/extract_snippets.py"


@pytest.fixture
def extractor():
    if not EXTRACTOR.exists():
        pytest.skip("Book-side extraction checks require the manuscript checkout")
    spec = importlib.util.spec_from_file_location("book_extract_under_test", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def printed_tree(extractor, snippet_id):
    snippets = extractor.extract_snippets(BOOK_ROOT / "companion/agentic-retail-foundations")
    specs = extractor.load_excerpts(EXTRACTOR.with_name("excerpts.json"))
    snippet = snippets[snippet_id]
    content = (
        extractor.apply_excerpt(snippet, specs[snippet_id])[0] if snippet_id in specs else snippet.content
    )
    return ast.parse(textwrap.dedent(content))


def test_eval_hook_is_a_complete_definition(extractor):
    tree = printed_tree(extractor, "eval-hook-demo")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "eval_hook" for node in tree.body)


def test_approval_fields_keep_their_class_ownership(extractor):
    tree = printed_tree(extractor, "hitl-approval-api-models")
    fields = {
        node.name: {field.target.id for field in node.body if isinstance(field, ast.AnnAssign)}
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    assert "requester" in fields["PendingReview"]
    assert "message" not in fields["PendingReview"]
    assert {"requester", "reviewer", "status", "timestamp"} <= fields["ReviewOutcome"]
    assert any(isinstance(node, ast.AnnAssign) and node.target.id == "pending_reviews" for node in tree.body)


def test_printed_schema_shows_the_advertised_validation_flow(extractor):
    tree = printed_tree(extractor, "structured-output-demo")
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"minor_units", "effective_window", "validate_tool_payload", "run_pipeline"} <= functions.keys()
    pipeline = ast.unparse(functions["run_pipeline"])
    assert pipeline.index("validate_tool_payload(payload)") < pipeline.index("apply_price(update)")
    assert "except ValidationError" in pipeline and "route_to_human_review" in pipeline


def test_complete_excerpt_rejects_dangling_constructor(extractor):
    snippet = extractor.Snippet("broken", Path("/companion/demo.py"), 1, 1, "run = AgentRun(\n", ".py")
    spec = {
        "source": "demo.py",
        "region_sha256": extractor.region_sha256(snippet.content),
        "plan": [[1, 1]],
        "python_syntax": "complete",
    }
    with pytest.raises(ValueError, match="Python syntax"):
        extractor.apply_excerpt(snippet, spec)


def test_cutting_a_complete_region_cannot_create_a_continuation(extractor):
    source = "def demo():\n    run = AgentRun(\n        trace_id='demo',\n    )\n"
    snippet = extractor.Snippet("cut", Path("/companion/demo.py"), 1, 4, source, ".py")
    spec = {
        "source": "demo.py",
        "region_sha256": extractor.region_sha256(source),
        "plan": [[1, 2], "# ... (fixture fields; see demo.py)"],
    }
    with pytest.raises(ValueError, match="Python syntax"):
        extractor.apply_excerpt(snippet, spec)
