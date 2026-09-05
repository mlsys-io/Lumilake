"""Source template for the observation LambdaOp.

This file is the literal source of the observation function that runs inside
the server's restricted Lambda sandbox. It is loaded as text and interpolated
(the ``{width}`` placeholder becomes the preview character budget) before being
embedded in the round graph's ``LambdaOp._code``.

The sandbox injects ``json`` as a global (never imported) and whitelists only a
few builtins. ``json.JSONDecodeError`` is attribute access on the whitelisted
``json`` module, so it is catchable inside the sandbox; malformed or plain-text
input is caught and treated as text rather than raising.
"""


def observe(args):
    # ``args`` is one entry per leaf. Each entry may be a JSON string, a list
    # of JSON strings (archived leaf outputs), a list of records, or plain
    # text. Normalize every entry into a flat list of records so the numeric
    # stats below see real values.
    rows = []
    for entry in args:
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)  # noqa: F821  # type: ignore[name-defined]
            except json.JSONDecodeError:  # noqa: F821  # type: ignore[name-defined]
                rows.append(entry)
                continue
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
                    except json.JSONDecodeError:  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
                        pass
                rows.append(item)
        else:
            rows.append(entry)
    # SQL retrievals arrive as column-oriented DataFrames: a list of dicts
    # mapping each column to a {row_index: value} map. Transpose to
    # row-oriented records so the numeric stats below see real values.
    if rows and isinstance(rows[0], dict):
        first_vals = list(rows[0].values())
        if first_vals and isinstance(first_vals[0], dict):
            records = {}
            for cmap in rows:
                if not isinstance(cmap, dict):
                    continue
                for col, idxvals in cmap.items():
                    if not isinstance(idxvals, dict):
                        continue
                    for idx, val in idxvals.items():
                        rec = records.get(idx)
                        if rec is None:
                            rec = {}
                            records[idx] = rec
                        rec[col] = val
            rows = list(records.values())
    n = len(rows)
    cols = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r:
            if isinstance(r[k], (int, float)) and k not in cols:
                cols.append(k)
    sums = {}
    mins = {}
    maxs = {}
    for k in cols:
        vals = [
            float(r[k])
            for r in rows
            if isinstance(r, dict) and k in r and isinstance(r[k], (int, float))
        ]
        if vals:
            sums[k] = sum(vals)
            mins[k] = min(vals)
            maxs[k] = max(vals)
    lines = [f"rows={n}"]
    for k in cols:
        if k in sums:
            cnt = len(
                [
                    1
                    for r in rows
                    if isinstance(r, dict) and k in r and isinstance(r[k], (int, float))
                ]
            )
            lines.append(
                f"{k}: n={cnt} sum={sums[k]:.4g} min={mins[k]:.4g} "
                f"max={maxs[k]:.4g} avg={sums[k] / max(1, cnt):.4g}"
            )
    preview = json.dumps(rows[:3], ensure_ascii=False)  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
    if len(preview) > {width}:  # noqa: F821  # type: ignore[name-defined]
        preview = preview[:{width}] + "..."  # noqa: F821  # type: ignore[name-defined]
    lines.append("preview=" + preview)
    return "\n".join(lines)
