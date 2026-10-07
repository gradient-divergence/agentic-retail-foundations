import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_order_demo_notebook_uses_marimo_state_setter(monkeypatch):
    cells = []

    class FakeApp:
        def __init__(self, **kwargs):
            pass

        def cell(self, function):
            cells.append(function)
            return function

    marimo = ModuleType("marimo")
    marimo.App = FakeApp
    monkeypatch.setitem(sys.modules, "marimo", marimo)
    path = Path(__file__).resolve().parents[2] / "notebooks/end-to-end-integration-for-autonomous-retail.py"
    spec = importlib.util.spec_from_file_location("integration_notebook_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    updates = []
    mo = SimpleNamespace(
        state=lambda _: (lambda: [], updates.append),
        ui=SimpleNamespace(button=lambda **_: "button"),
        md=lambda _: None,
    )
    state_cell = next(
        cell
        for cell in cells
        if list(inspect.signature(cell).parameters) == ["mo"]
        and "run_ord_orch_button" in cell.__code__.co_varnames
    )
    _, setter = state_cell(mo)
    assert callable(setter)
    runner_cell = next(
        cell for cell in cells if list(inspect.signature(cell).parameters) == ["set_ord_orch_logs"]
    )

    async def run_orchestration_simulation():
        pass

    fake_demo = ModuleType("demos.order_orchestration_demo")
    fake_demo.run_orchestration_simulation = run_orchestration_simulation
    monkeypatch.setitem(sys.modules, "demos.order_orchestration_demo", fake_demo)
    runner = runner_cell(setter)[0]
    asyncio.run(runner())
    assert updates == [
        ["Running Order Orchestration Demo... (Check console)"],
        ["Order Orchestration Demo Completed (check console)."],
    ]
