import asyncio
import inspect
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path to allow `import agents`, `import utils`, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    # This suite uses synchronous fixtures and one event loop per coroutine test.
    if not pyfuncitem.config.pluginmanager.hasplugin("asyncio") and inspect.iscoroutinefunction(
        pyfuncitem.obj
    ):
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in inspect.signature(pyfuncitem.obj).parameters
            if name in pyfuncitem.funcargs
        }
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None
