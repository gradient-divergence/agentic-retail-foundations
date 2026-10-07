import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_shelf_demo_stops_live_monitor_and_preserves_final_issues():
    source = Path("notebooks/core-technologies-enabling-agentic-retail.py").read_text()
    cell = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.AsyncFunctionDef) and child.name == "run_shelf_monitoring_demo"
            for child in node.body
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "notebook-cell", "exec"), namespace)

    class Shelf:
        detection_model = object()
        camera_streams = {"CAM01": "fake", "CAM02": "fake"}
        detected_issues = {}
        stopped = False

        async def start_monitoring_section(self, loc, section):
            self.detected_issues[section] = [{"type": "LOW_STOCK"}]
            while not self.stopped:
                await asyncio.sleep(0)

        async def stop_all_monitoring(self):
            self.stopped = True
            self.detected_issues.clear()

    async def short_sleep(duration):
        await asyncio.sleep(0)

    shelf = Shelf()
    fake_asyncio = SimpleNamespace(
        create_task=asyncio.create_task,
        gather=asyncio.gather,
        sleep=short_sleep,
    )
    mo = SimpleNamespace(
        md=lambda text: text,
        vstack=lambda parts: "\n".join(map(str, parts)),
        ui=SimpleNamespace(table=lambda data, **kwargs: str(data)),
    )
    (run_demo,) = namespace["_"](fake_asyncio, logging.getLogger(__name__), mo, pd, shelf)

    async def run():
        return await asyncio.wait_for(run_demo(), timeout=0.1)

    output = asyncio.run(run())
    assert shelf.stopped
    assert "LOW_STOCK" in output


def test_shelf_setup_without_model_or_camera_is_clean():
    source = Path("notebooks/core-technologies-enabling-agentic-retail.py").read_text()
    cell = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "model_path" for target in child.targets)
            for child in node.body
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "notebook-cell", "exec"), namespace)
    messages = []

    def missing_model(*args, **kwargs):
        raise ValueError("missing model")

    kwargs = {
        "DummyInventorySystem": lambda: None,
        "DummyPlanogramDB": lambda: None,
        "ShelfMonitoringAgent": missing_model,
        "mo": SimpleNamespace(md=messages.append),
        "os": SimpleNamespace(getenv=lambda key, default="": default),
    }
    setup = namespace["_"]
    args = {name: kwargs[name] for name in setup.__code__.co_varnames[: setup.__code__.co_argcount]}
    assert setup(**args) == (None,)
    assert "SHELF_MODEL_PATH" in messages[0]


def test_notebook_return_methods_reach_the_customer_prompt():
    from agents.response_builder import build_response_prompt

    source = Path("notebooks/core-technologies-enabling-agentic-retail.py").read_text()
    cell = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "policy_guidelines" for target in child.targets
            )
            for child in node.body
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "notebook-cell", "exec"), namespace)

    def agent_factory(product, order, customer, policy, key):
        return SimpleNamespace(policies=policy)

    (agent,) = namespace["_"](lambda: None, lambda: None, agent_factory, SimpleNamespace(environ={}))
    prompt = build_response_prompt(
        customer_info={},
        intent="return_request",
        message="Return?",
        conversation_history=[],
        context_data={"return_eligibility": {"eligible": True}, "return_policy": agent.policies["returns"]},
    )
    assert "Return Methods: Mail, Store" in prompt
