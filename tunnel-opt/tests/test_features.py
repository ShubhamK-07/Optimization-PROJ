import numpy as np

from src.features import (
    compute_slope,
    compute_roughness,
    compute_curvature,
)


def test_slope_on_planar_surface():
    """
    A planar surface z = ax + by + c
    must have constant slope sqrt(a^2 + b^2).
    """

    x = np.arange(7) * 10.0
    y = np.arange(7) * 10.0

    X, Y = np.meshgrid(x, y)

    a = 0.3
    b = 0.4

    elevation = a * X + b * Y + 100

    water_mask = np.ones(elevation.shape, dtype=bool)

    slope = compute_slope(elevation, x, y, water_mask)

    expected = np.sqrt(a**2 + b**2)

    np.testing.assert_allclose(
        slope[1:-1, 1:-1],
        expected,
        atol=1e-10
    )


def test_curvature_on_quadratic_surface():
    """
    For z = a(x^2 + y^2),
    the Laplacian is 4a.
    """

    x = np.arange(7) * 10.0
    y = np.arange(7) * 10.0

    X, Y = np.meshgrid(x, y)

    a = 0.002

    elevation = a * (X**2 + Y**2)

    water_mask = np.ones(elevation.shape, dtype=bool)

    curvature = compute_curvature(
        elevation,
        x,
        y,
        water_mask
    )

    expected = abs(4 * a)

    np.testing.assert_allclose(
        curvature[1:-1, 1:-1],
        expected,
        atol=1e-10
    )


def test_roughness_on_constant_slope():
    """
    A planar elevation surface produces
    constant slope and therefore zero roughness.
    """

    x = np.arange(7) * 10.0
    y = np.arange(7) * 10.0

    X, Y = np.meshgrid(x, y)

    elevation = 0.3 * X + 0.4 * Y

    water_mask = np.ones(elevation.shape, dtype=bool)

    slope = compute_slope(
        elevation,
        x,
        y,
        water_mask
    )

    roughness = compute_roughness(
        slope,
        water_mask
    )

    np.testing.assert_allclose(
        roughness[3, 3],
        0.0,
        atol=1e-10
    )