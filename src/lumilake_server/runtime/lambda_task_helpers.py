"""Worker-side helpers for generated standalone Lambda and reshape tasks."""

import json
import os


def _text(value):
    return value if isinstance(value, str) else json.dumps(value)


def _walk_path(value, parts, where):
    for idx, part in enumerate(parts):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
        if (
            isinstance(value, dict)
            and list(value) == ["df"]
            and isinstance(value["df"], str)
        ):
            try:
                table = json.loads(value["df"])
            except ValueError:
                raise ValueError(f"bad JSON at {where}")
            if part not in table:
                raise ValueError(f"missing {part!r} at {where}")
            column = table[part]
            if not isinstance(column, dict):
                raise ValueError(f"missing {part!r} at {where}")
            value = [column[k] for k in sorted(column, key=int)]
            rest = parts[idx + 1 :]
            if rest:
                return [_walk_path(v, rest, where) for v in value]
            return value
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError(f"missing {part!r} at {where}")
    return value


def _row_content(row):
    try:
        return row["json"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return row.get("text", row)


def _column(entry, inputs, text):
    if entry["kind"] == "literal":
        if "file" in entry:
            with open(entry["file"], encoding="utf-8") as fh:
                values = json.load(fh)
        else:
            values = entry["values"]
        return [text(v) for v in values]
    with open(os.path.join(inputs[entry["node"]], "results.json")) as fh:
        envelope = json.load(fh)
    result = envelope.get("result", envelope)
    items = result.get("items")
    if items is None and isinstance(result.get("value"), dict):
        items = result["value"].get("items")
    if items is None and isinstance(result.get("text"), str):
        items = [{"output": result["text"]}]
    if not isinstance(items, list):
        raise ValueError(f"upstream {entry['node']!r} produced no items")
    path = entry.get("path")
    if path:
        parts = path.split(".")
        return [
            _walk_path(i, parts, f"{entry['node']} item {idx} path {path}")
            for idx, i in enumerate(items)
        ]
    if entry.get("flatten"):
        flat = []
        for i in items:
            if isinstance(i, dict) and "rows" in i:
                for row in i["rows"]:
                    flat.append(text(_row_content(row)))
            else:
                out = i["output"] if isinstance(i, dict) and "output" in i else i
                if isinstance(out, list):
                    flat.extend(text(v) for v in out)
                else:
                    flat.append(text(out))
        return flat
    return [
        text(i["output"]) if isinstance(i, dict) and "output" in i else text(i)
        for i in items
    ]
