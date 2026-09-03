"""Minimal JSON schema validator shared by anything that parses LLM JSON output.

No external deps (not even ``jsonschema``) — only handles type/required/
minLength, which is enough to catch the common LLM mistakes (missing field,
empty string, wrong top-level type). Used by M3's hypothesis parser, the
agentic loop's action parser, and failure-mode cluster naming.
"""

from __future__ import annotations


def validate_json_shape(data: object, schema: dict) -> list[str]:
    """Returns a list of error strings; empty means valid."""
    errors: list[str] = []

    def _check(node: object, s: dict, path: str) -> None:
        expected_type = s.get("type")
        if expected_type == "array":
            if not isinstance(node, list):
                errors.append(f"{path}: expected array, got {type(node).__name__}")
                return
            item_schema = s.get("items", {})
            for i, item in enumerate(node):
                _check(item, item_schema, f"{path}[{i}]")
        elif expected_type == "object":
            if not isinstance(node, dict):
                errors.append(f"{path}: expected object, got {type(node).__name__}")
                return
            for req in s.get("required", []):
                if req not in node:
                    errors.append(f"{path}: missing required field '{req}'")
            for prop, prop_schema in s.get("properties", {}).items():
                if prop in node:
                    _check(node[prop], prop_schema, f"{path}.{prop}")
        elif expected_type == "string":
            if not isinstance(node, str):
                errors.append(f"{path}: expected string, got {type(node).__name__}")
            elif "minLength" in s and len(node) < s["minLength"]:
                errors.append(
                    f"{path}: string too short ({len(node)} < {s['minLength']})"
                )

    _check(data, schema, "$")
    return errors
