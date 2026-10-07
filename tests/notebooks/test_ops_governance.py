"""Exercise affected notebook cells without requiring the marimo UI runtime."""

import ast
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.demos.api_fakes import load_demo


def notebook_cells(monkeypatch, path):
    class App:
        def __init__(self, **kwargs):
            self.cells = []

        def cell(self, function):
            self.cells.append(function)
            return function

        def class_definition(self, cls):
            return cls

    monkeypatch.setitem(sys.modules, "marimo", SimpleNamespace(App=App))
    return runpy.run_path(path)["app"].cells


def test_notebook_inventory_includes_forecast_only_skus():
    source = Path("notebooks/implementing-agentic-systems-in-retail.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InventoryAgent")
    cls.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "notebook", "exec"), namespace)
    assert namespace["InventoryAgent"](10).evaluate_restock({}, {"new": 15}) == {"new": 25}


def test_governance_notebook_reuses_approval_api(monkeypatch):
    demo = load_demo(monkeypatch, "hitl_approval_api_demo")
    cells = notebook_cells(monkeypatch, "notebooks/ethical-considerations-and-governance.py")
    assert cells[3]() == (demo.app,)


def test_monitoring_notebook_initializes_without_database_package(monkeypatch):
    demo = load_demo(monkeypatch, "monitoring_dashboard_supabase_demo")
    monkeypatch.setitem(sys.modules, "supabase", None)
    cells = notebook_cells(monkeypatch, "notebooks/implementing-agentic-systems-in-retail.py")
    assert cells[0]()
    assert cells[1]() == (demo.app,)


def test_case_studies_initialize_without_openai_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    cells = notebook_cells(monkeypatch, "notebooks/real-world-case-studies.py")
    _, assistant, inventory, pricing = cells[0]()
    assert assistant.__module__ == "demos.virtual_shopping_responses_demo"
    assert callable(inventory) and callable(pricing)
