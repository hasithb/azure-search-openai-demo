"""Validate a v4 evidence JSON document against a local JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, RefResolver


class SchemaContractError(ValueError):
    """Raised when a v4 report does not satisfy its schema."""


def validate_document(document: dict[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    resolver = RefResolver.from_schema(schema)
    for sibling_path in schema_path.parent.glob("*.schema.json"):
        sibling_schema = json.loads(sibling_path.read_text(encoding="utf-8"))
        schema_id = sibling_schema.get("$id")
        if schema_id:
            resolver.store[schema_id] = sibling_schema
    errors = sorted(Draft202012Validator(schema, resolver=resolver).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "$"
        raise SchemaContractError(f"{schema_path.name}: {location}: {errors[0].message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise SchemaContractError("v4 schema input must be a JSON object")
        validate_document(document, args.schema)
    except (OSError, json.JSONDecodeError, SchemaContractError) as error:
        print(f"Schema validation failed: {error}")
        return 1
    print(f"Schema validation passed: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
