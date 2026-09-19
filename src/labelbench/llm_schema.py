"""Constrain each overlap group to exactly one decision during generation."""

from __future__ import annotations

from typing import Any

STRUCTURED_PROMPT = '''Review document segmentation against the image, not just confidence scores.
Input: [group_id, [[provider,label,confidence,polygon],...]]. Coordinates are normalized 0..1.
Return {"groups":{"GROUP_ID":decision,...},"add":[],"notes":""}.
Every supplied group ID must occur exactly once in groups.
A decision is ONE zero-based candidate index, "drop", "uncertain", or a new polygon.
Choose the candidate whose boundary best fits the visible text. Candidates are ALTERNATIVES.
Never accept all candidates from a group. Do not always choose index 0 or the highest confidence.
Use "drop" only if no candidate describes a real object; "uncertain" if the image is unclear.
A new polygon (3..50 [x,y] points in 0..1) replaces the entire group with one improved boundary.
For a valid singleton choose 0. No bounding boxes, OCR text, markdown, or long explanations.
This is one batch of a larger page review. Objects absent from this batch may exist in other batches.
Do not invent missing objects from their absence in this batch; leave add empty for this pass.
Example: group 7 has three alternatives, option 2 fits best => "groups":{"7":2}.
Close the JSON and stop.'''


def response_format(groups: list[Any]) -> dict[str, Any]:
    polygon = {
        "type": "array", "minItems": 3, "maxItems": 50,
        "items": {"type": "array", "minItems": 2, "maxItems": 2,
                  "items": {"type": "number", "minimum": 0, "maximum": 1}},
    }
    properties = {
        str(group_id): {"anyOf": [
            {"type": "integer", "minimum": 0, "maximum": len(options) - 1},
            {"type": "string", "enum": ["drop", "uncertain"]}, polygon,
        ]}
        for group_id, options in groups
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "groups": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
            "add": {"type": "array", "maxItems": 0, "items": {"type": "string"}},
            "notes": {"type": "string", "maxLength": 240},
        },
        "required": ["groups", "add", "notes"],
    }
    return {"type": "json_schema", "json_schema": {
        "name": "polygon_review", "strict": True, "schema": schema,
    }}


def normalize_decisions(payload: dict[str, Any]) -> dict[str, Any]:
    if "groups" not in payload:
        return payload
    if not isinstance(payload["groups"], dict):
        raise TypeError("groups must be an object")
    result: dict[str, Any] = {"pick": [], "fuse": [], "drop": [], "uncertain": [],
                              "add": [], "notes": payload.get("notes", "")}
    for key, decision in payload["groups"].items():
        group_id = int(key)
        if type(decision) is int:
            result["pick"].append([group_id, decision])
        elif isinstance(decision, str) and decision in ("drop", "uncertain"):
            result[decision].append(group_id)
        elif isinstance(decision, list):
            result["fuse"].append([group_id, decision])
        else:
            raise ValueError("Unknown group decision")
    return result
