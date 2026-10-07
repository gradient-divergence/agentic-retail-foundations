import json
from pathlib import Path


def test_scenarios_match_schema_and_golden_actions():
    import jsonschema

    schema = json.loads(Path("evals/schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    scenarios = json.loads(Path("evals/scenarios.json").read_text())
    by_id = {case["id"]: case for case in scenarios}
    assert len(by_id) == len(scenarios)
    for case in scenarios:
        jsonschema.validate(case, schema)
    traces = json.loads(Path("evals/golden_traces.json").read_text())
    for trace in traces:
        jsonschema.validate(trace, {"$ref": "#/$defs/golden_trace", "$defs": schema["$defs"]})
        case = by_id[trace["case_id"]]
        assert trace["events"] == case["expected_events"]
        actions = [trace["decision"]] + [
            call["name"] for call in trace["tool_calls"] if call["name"] in case["expected_actions"]
        ]
        assert actions == case["expected_actions"]
