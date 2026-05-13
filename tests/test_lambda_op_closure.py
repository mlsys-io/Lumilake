import textwrap

from lumilake.ops import LambdaOp


def _make_quote_fn(payload: object):
    def fn(inputs: tuple[str, ...]) -> str:  # pragma: no cover - serialized
        return str(payload)

    return fn


def _make_backref_fn(payload: str):
    def fn(inputs: tuple[str, ...]) -> str:  # pragma: no cover - serialized
        return str(payload)

    return fn


def test_closure_with_mixed_quotes_does_not_break_repr_substitution() -> None:
    payload = {"k": '"SoundOfTexture\' on a Prayer" is a song title'}
    op = LambdaOp(inputs=[["x"]], fn=_make_quote_fn(payload))
    assert "SoundOfTexture" in op.code
    ns: dict = {}
    exec(textwrap.dedent(op.code), ns)  # noqa: S102
    assert "fn" in ns


def test_closure_with_digit_heavy_repr_does_not_inject_backreference() -> None:
    payload = "\\1 looks like a backref"
    op = LambdaOp(inputs=[["x"]], fn=_make_backref_fn(payload))
    assert repr(payload) in op.code
