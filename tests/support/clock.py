"""Deterministic virtual clock for scheduling tests.

The clock never sleeps on wall time; it only advances by explicit ``advance``
calls. All time measured on this clock is fully reproducible from a seed,
unlike ad hoc ``time.time`` / ``asyncio.sleep`` monkeypatching.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class VirtualClock:
    """A monotonic virtual clock starting at zero."""

    _now: float = 0.0

    def now(self) -> float:
        """Return the current virtual time."""
        return self._now

    def advance(self, dt: float) -> None:
        """Advance the clock by ``dt`` seconds (must be non-negative)."""
        if dt < 0:
            raise ValueError(f"cannot advance the clock backwards: {dt}")
        self._now += dt

    def monotonic(self) -> float:
        """Return a strictly monotonic reading (identical to ``now`` here)."""
        return self._now
