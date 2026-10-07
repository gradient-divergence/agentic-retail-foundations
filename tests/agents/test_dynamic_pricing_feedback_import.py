import subprocess
import sys


def test_module_import_does_not_require_streaming_clients():
    code = """
import sys
sys.modules['redis'] = None
sys.modules['kafka'] = None
from agents.dynamic_pricing_feedback import DynamicPricingAgent
assert callable(DynamicPricingAgent)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
