"""Exponentially-decayed attained resource-area, keyed by principal."""

import time
from collections.abc import Callable


class AttainedService:
    """Exponentially-decayed attained resource-area, keyed by principal.

    Each principal's value decays toward zero with a half-life and grows by the
    area of each charged round. Decay is applied lazily on read/write (the
    ``(value, last_update)`` pair is stored, no timer runs), so the object is
    safe to hold across arbitrarily long gaps.
    """

    def __init__(
        self,
        half_life_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be > 0")
        self._half_life = half_life_seconds
        self._clock = clock
        self._values: dict[str, tuple[float, float]] = {}

    def _decay(self, value: float, last_update: float, now: float) -> float:
        dt = max(0.0, now - last_update)
        return value * 0.5 ** (dt / self._half_life)

    def charge(self, key: str, area: float) -> None:
        now = self._clock()
        value, last_update = self._values.get(key, (0.0, now))
        value = self._decay(value, last_update, now)
        self._values[key] = (value + area, now)

    def get(self, key: str) -> float:
        now = self._clock()
        value, last_update = self._values.get(key, (0.0, now))
        return self._decay(value, last_update, now)


__all__ = ["AttainedService"]
