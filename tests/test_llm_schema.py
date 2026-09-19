from labelbench.llm import _parse_json
from labelbench.llm_schema import normalize_decisions, response_format


def test_response_schema_requires_each_group_and_limits_indices() -> None:
    schema = response_format([[7, [[], [], []]], [12, [[]]]])["json_schema"]["schema"]
    groups = schema["properties"]["groups"]
    assert groups["required"] == ["7", "12"]
    assert groups["additionalProperties"] is False
    assert groups["properties"]["7"]["anyOf"][0]["maximum"] == 2
    assert groups["properties"]["12"]["anyOf"][0]["maximum"] == 0


def test_structured_choices_preserve_geometry_and_statuses() -> None:
    polygon = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.2]]
    result = normalize_decisions({"groups": {"1": 2, "2": "drop", "3": "uncertain", "4": polygon}})
    assert result["pick"] == [[1, 2]]
    assert result["drop"] == [2]
    assert result["uncertain"] == [3]
    assert result["fuse"] == [[4, polygon]]


def test_structured_parser_rejects_wrong_group_types() -> None:
    assert _parse_json('{"groups":{"0":0}}')["pick"] == [[0, 0]]
    assert _parse_json('{"groups":{"0":true}}') is None
    assert _parse_json('{"groups":{"not-an-id":0}}') is None
    assert _parse_json('{"groups":[]}') is None
    assert _parse_json('{"groups":{"0":0,"0":1}}') is None
