"""
Event profile classes for defining value distribution along linear events.

Profiles describe how an event's value or influence is distributed over its
length. They operate on normalized positions t ∈ [0, 1] where 0 is the event's
beginning and 1 is the event's end.
"""

from __future__ import annotations

import numpy as np


class EventProfile:
    """
    Base class for event value profiles.

    Subclasses must implement `_integral(a, b)` which returns the definite
    integral of the profile function over [a, b] ⊂ [0, 1], normalized so
    that the full integral over [0, 1] equals 1.0.
    """

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the profile at normalized positions.

        Parameters
        ----------
        t : array-like
            Positions in [0, 1] where 0 is event start and 1 is event end.

        Returns
        -------
        np.ndarray
            Profile values at the given positions.

        Raises
        ------
        ValueError
            If any values in t are outside [0, 1].
        """
        t = np.asarray(t, dtype=float)
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(
                "Profile positions must be in [0, 1]."
            )
        return self._evaluate(t)

    def integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the definite integral of the profile over [a, b].

        The result is normalized so that integral(0, 1) = 1.0 for all profiles.

        Parameters
        ----------
        a : array-like
            Start positions in [0, 1].
        b : array-like
            End positions in [0, 1].

        Returns
        -------
        np.ndarray
            Integral values, each in [0, 1].

        Raises
        ------
        ValueError
            If any values in a or b are outside [0, 1].
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(b > 1):
            raise ValueError(
                "Integral bounds must be in [0, 1]."
            )
        return self._integral(a, b)

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class UniformProfile(EventProfile):
    """
    Uniform (flat) profile — equal value distribution across the event.

    This reproduces the default behavior where all parts of an event
    contribute equally.

    Profile function: f(t) = 1
    """

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        return np.ones_like(t)

    def _integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return b - a


class TriangularProfile(EventProfile):
    """
    Triangular profile — peak at center, zero at edges.

    Value is concentrated at the event's center and tapers linearly to zero
    at both ends. Useful for point-derived linear events where the true
    location is at the center.

    Profile function: f(t) = 4t for t < 0.5, 4(1-t) for t >= 0.5
    (normalized so total area = 1)
    """

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 4.0 * t, 4.0 * (1.0 - t))

    def _integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Antiderivative F(t):
        #   t < 0.5: F(t) = 2t²
        #   t >= 0.5: F(t) = -2t² + 4t - 1
        # integral(a, b) = F(b) - F(a)
        return self._antideriv(b) - self._antideriv(a)

    @staticmethod
    def _antideriv(t: np.ndarray) -> np.ndarray:
        return np.where(t < 0.5, 2.0 * t**2, -2.0 * t**2 + 4.0 * t - 1.0)


class ParabolicProfile(EventProfile):
    """
    Parabolic profile — smooth peak at center, zero at edges.

    Similar to triangular but with a smooth (differentiable) curve.
    No sharp kink at the peak.

    Profile function: f(t) = 6t(1-t)
    (normalized so total area = 1)
    """

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        return 6.0 * t * (1.0 - t)

    def _integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Antiderivative of 6t(1-t) = 6t - 6t²:
        #   F(t) = 3t² - 2t³
        # integral(a, b) = F(b) - F(a)
        return self._antideriv(b) - self._antideriv(a)

    @staticmethod
    def _antideriv(t: np.ndarray) -> np.ndarray:
        return 3.0 * t**2 - 2.0 * t**3


class TrapezoidalProfile(EventProfile):
    """
    Trapezoidal profile — flat center with linear ramps at edges.

    Generalizes between uniform (ramp=0) and triangular (ramp=0.5).
    The `ramp` parameter specifies the fraction of each side that ramps
    from zero to full value.

    Parameters
    ----------
    ramp : float, default 0.25
        Fraction of the event length on each side that ramps linearly.
        Must satisfy 0 < ramp <= 0.5. A value of 0.25 means 25% ramp on
        each end with 50% flat in the middle.

    Profile function (for ramp=r):
        f(t) = t/r           for t < r
        f(t) = 1             for r <= t <= 1-r
        f(t) = (1-t)/r       for t > 1-r
    (scaled so total area = 1)
    """

    def __init__(self, ramp: float = 0.25) -> None:
        if not (0 < ramp <= 0.5):
            raise ValueError(
                f"ramp must satisfy 0 < ramp <= 0.5, got {ramp}"
            )
        self.ramp = ramp
        # Total area of unnormalized trapezoid: 1 - ramp
        # Scale factor to normalize area to 1: 1 / (1 - ramp)
        self._scale = 1.0 / (1.0 - ramp)

    def _evaluate(self, t: np.ndarray) -> np.ndarray:
        r = self.ramp
        result = np.where(
            t < r,
            t / r,
            np.where(t > 1.0 - r, (1.0 - t) / r, 1.0),
        )
        return result * self._scale

    def _integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._antideriv(b) - self._antideriv(a)

    def _antideriv(self, t: np.ndarray) -> np.ndarray:
        r = self.ramp
        s = self._scale
        # Piecewise antiderivative (continuous):
        #   t < r:         F(t) = s * t² / (2r)
        #   r <= t <= 1-r: F(t) = s * (t - r/2)
        #   t > 1-r:       F(t) = s * (1 - r - (1-t)² / (2r))
        return np.where(
            t < r,
            s * t**2 / (2.0 * r),
            np.where(
                t <= 1.0 - r,
                s * (t - r / 2.0),
                s * (1.0 - r - (1.0 - t) ** 2 / (2.0 * r)),
            ),
        )

    def __repr__(self) -> str:
        return f"TrapezoidalProfile(ramp={self.ramp})"


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

_PROFILE_REGISTRY: dict[str, type[EventProfile]] = {
    "uniform": UniformProfile,
    "triangular": TriangularProfile,
    "parabolic": ParabolicProfile,
    "trapezoidal": TrapezoidalProfile,
}


def resolve_profile(profile) -> EventProfile | None:
    """
    Resolve a profile specification to an EventProfile instance.

    Parameters
    ----------
    profile : None, str, or EventProfile
        - None: no profiling (returns None)
        - str: looked up in the profile registry and instantiated with defaults
        - EventProfile instance: returned as-is

    Returns
    -------
    EventProfile or None

    Raises
    ------
    TypeError
        If profile is not None, str, or EventProfile.
    ValueError
        If profile string is not recognized.
    """
    if profile is None:
        return None
    if isinstance(profile, EventProfile):
        return profile
    if isinstance(profile, str):
        key = profile.lower()
        if key not in _PROFILE_REGISTRY:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available profiles: {list(_PROFILE_REGISTRY.keys())}"
            )
        return _PROFILE_REGISTRY[key]()
    raise TypeError(
        f"profile must be None, a string, or an EventProfile instance, "
        f"got {type(profile).__name__}"
    )
