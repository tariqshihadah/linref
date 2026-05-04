"""
Unit tests for linref.events.profile module.

Tests cover profile classes, the resolve_profile function, and integration
with the overlay function.
"""

import unittest
import numpy as np
from linref.events.base import EventsData
from linref.events.profile import (
    EventProfile,
    UniformProfile,
    TriangularProfile,
    ParabolicProfile,
    TrapezoidalProfile,
    resolve_profile,
    _PROFILE_REGISTRY,
)
from linref.events.relate import overlay


ALL_PROFILES = [
    UniformProfile(),
    TriangularProfile(),
    ParabolicProfile(),
    TrapezoidalProfile(ramp=0.25),
    TrapezoidalProfile(ramp=0.5),
]


class TestEventProfileBase(unittest.TestCase):
    """Test the base EventProfile class."""

    def test_evaluate_not_implemented(self):
        """Base class _evaluate raises NotImplementedError."""
        p = EventProfile()
        with self.assertRaises(NotImplementedError):
            p(np.array([0.5]))

    def test_integral_not_implemented(self):
        """Base class _integral raises NotImplementedError."""
        p = EventProfile()
        with self.assertRaises(NotImplementedError):
            p.integral(np.array(0.0), np.array(1.0))

    def test_evaluate_out_of_bounds_raises(self):
        """Evaluate with values outside [0, 1] raises ValueError."""
        p = UniformProfile()
        with self.assertRaises(ValueError):
            p(np.array([-0.1, 0.5]))
        with self.assertRaises(ValueError):
            p(np.array([0.5, 1.1]))

    def test_integral_out_of_bounds_raises(self):
        """Integral with bounds outside [0, 1] raises ValueError."""
        p = TriangularProfile()
        with self.assertRaises(ValueError):
            p.integral(np.array(-0.1), np.array(0.5))
        with self.assertRaises(ValueError):
            p.integral(np.array(0.0), np.array(1.1))
        with self.assertRaises(ValueError):
            p.integral(np.array(-0.5), np.array(1.5))


class TestAllProfilesCommon(unittest.TestCase):
    """Tests that must hold for every profile."""

    def test_full_integral_equals_one(self):
        for p in ALL_PROFILES:
            with self.subTest(profile=repr(p)):
                np.testing.assert_allclose(
                    p.integral(np.array(0.0), np.array(1.0)), 1.0, atol=1e-12)

    def test_half_integrals_equal_point_five(self):
        for p in ALL_PROFILES:
            with self.subTest(profile=repr(p)):
                np.testing.assert_allclose(
                    p.integral(np.array(0.0), np.array(0.5)), 0.5, atol=1e-12)
                np.testing.assert_allclose(
                    p.integral(np.array(0.5), np.array(1.0)), 0.5, atol=1e-12)

    def test_zero_width_integral_equals_zero(self):
        a = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        for p in ALL_PROFILES:
            with self.subTest(profile=repr(p)):
                np.testing.assert_allclose(p.integral(a, a), 0.0, atol=1e-12)

    def test_2d_array_broadcasting(self):
        """Integral works with 2D arrays as used in overlay."""
        a = np.array([[0.0, 0.0], [0.5, 0.5]])
        b = np.array([[0.5, 1.0], [1.0, 1.0]])
        for p in ALL_PROFILES:
            with self.subTest(profile=repr(p)):
                result = p.integral(a, b)
                self.assertEqual(result.shape, (2, 2))
                self.assertTrue(np.all(result >= -1e-12))
                self.assertTrue(np.all(result <= 1.0 + 1e-12))


class TestUniformProfile(unittest.TestCase):
    """Test UniformProfile-specific behavior."""

    def setUp(self):
        self.p = UniformProfile()

    def test_evaluate_constant_one(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(self.p(t), [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_integral_quarters(self):
        a = np.array([0.0, 0.25, 0.5, 0.75])
        b = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            self.p.integral(a, b), [0.25, 0.25, 0.25, 0.25])


class TestTriangularProfile(unittest.TestCase):
    """Test TriangularProfile-specific behavior."""

    def setUp(self):
        self.p = TriangularProfile()

    def test_evaluate(self):
        # f(t) = 4t for t<0.5, 4(1-t) for t>=0.5
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(self.p(t), [0.0, 1.0, 2.0, 1.0, 0.0])

    def test_integral_center_half(self):
        # F(0.75) - F(0.25) = 0.875 - 0.125 = 0.75
        np.testing.assert_allclose(
            self.p.integral(np.array(0.25), np.array(0.75)), 0.75)

    def test_integral_first_quarter(self):
        # F(0.25) = 2*(0.25)^2 = 0.125
        np.testing.assert_allclose(
            self.p.integral(np.array(0.0), np.array(0.25)), 0.125)


class TestParabolicProfile(unittest.TestCase):
    """Test ParabolicProfile-specific behavior."""

    def setUp(self):
        self.p = ParabolicProfile()

    def test_evaluate(self):
        # f(t) = 6t(1-t)
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = [0.0, 6*0.25*0.75, 6*0.5*0.5, 6*0.75*0.25, 0.0]
        np.testing.assert_allclose(self.p(t), expected)

    def test_integral_center_half(self):
        # F(0.75) - F(0.25) = 0.84375 - 0.15625 = 0.6875
        np.testing.assert_allclose(
            self.p.integral(np.array(0.25), np.array(0.75)), 0.6875)

    def test_integral_first_quarter(self):
        # F(0.25) = 3*(0.0625) - 2*(0.015625) = 0.15625
        np.testing.assert_allclose(
            self.p.integral(np.array(0.0), np.array(0.25)), 0.15625)


class TestTrapezoidalProfile(unittest.TestCase):
    """Test TrapezoidalProfile-specific behavior across multiple ramp values."""

    ramp_values = [0.1, 0.25, 0.4, 0.5]

    def test_evaluate_ramp_region(self):
        """In [0, r]: f(t) = t / (r * (1-r))"""
        for r in self.ramp_values:
            p = TrapezoidalProfile(ramp=r)
            # Scale factor: normalizes total area to 1, also the plateau height
            s = 1.0 / (1.0 - r)
            t = np.array([0.0, r / 2, r])
            expected = np.array([0.0, s * 0.5, s])
            with self.subTest(ramp=r):
                np.testing.assert_allclose(p(t), expected)

    def test_evaluate_flat_region(self):
        """In [r, 1-r]: f(t) = 1/(1-r) (constant)."""
        for r in self.ramp_values:
            p = TrapezoidalProfile(ramp=r)
            # Scale factor: normalizes total area to 1, also the plateau height
            s = 1.0 / (1.0 - r)
            # Sample midpoints of the flat region (skip r=0.5 which has no flat)
            if r < 0.5:
                t = np.array([r, 0.5, 1.0 - r])
                with self.subTest(ramp=r):
                    np.testing.assert_allclose(p(t), [s, s, s])

    def test_integral_ramp_only(self):
        """integral(0, r) = r / (2*(1-r)) for all r."""
        for r in self.ramp_values:
            p = TrapezoidalProfile(ramp=r)
            expected = r / (2.0 * (1.0 - r))
            with self.subTest(ramp=r):
                np.testing.assert_allclose(
                    p.integral(np.array(0.0), np.array(r)), expected)

    def test_validation_ramp_invalid(self):
        for ramp in [0.0, -0.1, 0.6]:
            with self.subTest(ramp=ramp):
                with self.assertRaises(ValueError):
                    TrapezoidalProfile(ramp=ramp)


class TestResolveProfile(unittest.TestCase):
    """Test the resolve_profile function."""

    def test_none_returns_none(self):
        self.assertIsNone(resolve_profile(None))

    def test_instance_passthrough(self):
        p = TriangularProfile()
        self.assertIs(resolve_profile(p), p)

    def test_string_lookup(self):
        for name, cls in _PROFILE_REGISTRY.items():
            with self.subTest(name=name):
                self.assertIsInstance(resolve_profile(name), cls)

    def test_string_case_insensitive(self):
        self.assertIsInstance(resolve_profile("Triangular"), TriangularProfile)

    def test_unknown_string_raises(self):
        with self.assertRaises(ValueError):
            resolve_profile("unknown_profile")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_profile(42)


class TestProfileOverlayIntegration(unittest.TestCase):
    """Integration tests: profile with the overlay function."""

    def setUp(self):
        self.left = EventsData(
            begs=np.array([0.0, 5.0, 15.0]),
            ends=np.array([10.0, 15.0, 25.0]),
        )
        self.right = EventsData(
            begs=np.array([0.0, 10.0]),
            ends=np.array([10.0, 20.0]),
        )

    def _to_dense(self, result):
        return result.toarray() if hasattr(result, 'toarray') else np.asarray(result)

    def test_uniform_equals_standard(self):
        """profile='uniform' produces identical result to profile=None."""
        standard = self._to_dense(
            overlay(self.left, self.right, normalize=True, norm_by='right'))
        profiled = self._to_dense(
            overlay(self.left, self.right, normalize=True, norm_by='right',
                    profile='uniform'))
        np.testing.assert_allclose(profiled, standard, atol=1e-12)

    def test_full_overlap_equals_one(self):
        """Full overlap of one event yields weight=1.0 for all profiles."""
        left = EventsData(begs=np.array([0.0]), ends=np.array([10.0]))
        right = EventsData(begs=np.array([0.0]), ends=np.array([10.0]))
        for name in [None, 'triangular', 'parabolic', 'trapezoidal']:
            with self.subTest(profile=name):
                result = self._to_dense(
                    overlay(left, right, normalize=True, norm_by='right',
                            profile=name))
                np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-12)

    def test_symmetric_halves_sum_to_one(self):
        """Two left events each covering exactly half of a right event sum to 1."""
        result = self._to_dense(
            overlay(self.left, self.right, normalize=True, norm_by='right',
                    profile='triangular'))
        np.testing.assert_allclose(result[:, 1].sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(result[1, 1], 0.5, atol=1e-12)
        np.testing.assert_allclose(result[2, 1], 0.5, atol=1e-12)

    def test_no_overlap_is_zero(self):
        """Non-overlapping events produce zero weight."""
        for name in ['triangular', 'parabolic', 'trapezoidal']:
            with self.subTest(profile=name):
                result = self._to_dense(
                    overlay(self.left, self.right, normalize=True,
                            norm_by='right', profile=name))
                self.assertEqual(result[2, 0], 0.0)

    def test_grouped_events(self):
        """Profile respects group boundaries."""
        left = EventsData(
            begs=np.array([0.0, 0.0]), ends=np.array([10.0, 10.0]),
            groups=np.array([1, 2]))
        right = EventsData(
            begs=np.array([0.0, 0.0]), ends=np.array([10.0, 10.0]),
            groups=np.array([1, 2]))
        result = self._to_dense(
            overlay(left, right, normalize=True, norm_by='right',
                    profile='parabolic'))
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-12)
        self.assertEqual(result[0, 1], 0.0)
        self.assertEqual(result[1, 0], 0.0)

    def test_norm_by_left(self):
        """Profile works with norm_by='left'."""
        result = self._to_dense(
            overlay(self.left, self.right, normalize=True, norm_by='left',
                    profile='triangular'))
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-12)

    def test_aggregation_sum(self):
        """sum() through EventsRelation with profile produces correct result."""
        left = EventsData(
            begs=np.array([0.0, 10.0, 20.0]),
            ends=np.array([10.0, 20.0, 30.0]))
        right = EventsData(
            begs=np.array([0.0, 10.0, 20.0]),
            ends=np.array([10.0, 20.0, 30.0]))
        rel = left.relate(right)
        data = np.array([1.0, 2.0, 3.0])
        result = rel.sum(data=data, profile='triangular')
        np.testing.assert_allclose(result, data, atol=1e-12)

    def test_aggregation_mean(self):
        """mean() through EventsRelation with profile produces correct result."""
        left = EventsData(
            begs=np.array([0.0, 10.0, 20.0]),
            ends=np.array([10.0, 20.0, 30.0]))
        right = EventsData(
            begs=np.array([0.0, 10.0, 20.0]),
            ends=np.array([10.0, 20.0, 30.0]))
        rel = left.relate(right)
        data = np.array([4.0, 4.0, 4.0])
        result = rel.mean(data=data, profile='parabolic')
        np.testing.assert_allclose(result, data, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
