"""Small route fakes for exercising demo logic without the optional web stack."""

import importlib
import sys
from types import SimpleNamespace


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FastAPI:
    def get(self, *args, **kwargs):
        return lambda function: function

    post = get
    middleware = get


class Response:
    def __init__(self, content, media_type):
        self.body = content
        self.media_type = media_type


def load_demo(monkeypatch, name):
    monkeypatch.setitem(
        sys.modules,
        "fastapi",
        SimpleNamespace(FastAPI=FastAPI, HTTPException=HTTPException, Request=object, Response=Response),
    )
    package = importlib.import_module("demos")
    monkeypatch.setattr(package, name, None, raising=False)
    module_name = f"demos.{name}"
    # Track the previously absent entry too, so teardown removes the fake app.
    monkeypatch.setitem(sys.modules, module_name, None)
    del sys.modules[module_name]
    return importlib.import_module(module_name)
