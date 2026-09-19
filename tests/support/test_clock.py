"""Tests of the deterministic virtual clock used by scheduling tests."""

import pytest

from tests.support.clock import VirtualClock


def test_clock_is_deterministic_and_never_sleeps() -> None:
    clock = VirtualClock()
    assert clock.now() == 0.0
    clock.advance(1.5)
    clock.advance(2.5)
    assert clock.now() == 4.0
    assert clock.monotonic() == 4.0
    # Advancing is pure arithmetic; it must not depend on wall time.
    before = clock.now()
    clock.advance(0.0)
    assert clock.now() == before


def test_clock_rejects_backwards_advance() -> None:
    clock = VirtualClock()
    clock.advance(5.0)
    with pytest.raises(ValueError):
        clock.advance(-1.0)
