import numpy as np
from scipy.ndimage import generic_filter

def compute_slope(elevation, x, y, water_mask):
    """
    Compute terrain slope using central differences.

    Slope is calculated only where the centre cell and
    its four axial neighbours are valid water cells.
    """

    elevation = np.asarray(elevation, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)

    rows, cols = elevation.shape

    slope = np.full((rows, cols), np.nan, dtype=float)

    # Central differences require an interior cell.
    dz_dx = (
        elevation[1:-1, 2:] - elevation[1:-1, :-2]
    ) / (x[2:] - x[:-2])[None, :]

    dz_dy = (
        elevation[2:, 1:-1] - elevation[:-2, 1:-1]
    ) / (y[2:] - y[:-2])[:, None]

    # All five cells in the stencil must be water.
    valid = (
        water_mask[1:-1, 1:-1]
        & water_mask[1:-1, :-2]
        & water_mask[1:-1, 2:]
        & water_mask[:-2, 1:-1]
        & water_mask[2:, 1:-1]
    )

    # Ensure the elevation values are finite.
    valid &= (
        np.isfinite(elevation[1:-1, 1:-1])
        & np.isfinite(elevation[1:-1, :-2])
        & np.isfinite(elevation[1:-1, 2:])
        & np.isfinite(elevation[:-2, 1:-1])
        & np.isfinite(elevation[2:, 1:-1])
    )

    interior_slope = np.sqrt(dz_dx**2 + dz_dy**2)

    slope[1:-1, 1:-1] = np.where(
        valid,
        interior_slope,
        np.nan
    )

    return slope

def compute_roughness(slope, water_mask, window_size=3):
    """
    Compute terrain roughness as local standard deviation of slope.

    Invalid slope values are ignored.
    A minimum of 5 valid values is required in each window.
    """

    slope = np.asarray(slope, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    def local_std(values):
        valid_values = values[np.isfinite(values)]

        if len(valid_values) < 5:
            return np.nan

        return np.std(valid_values, ddof=0)

    roughness = generic_filter(
        slope,
        function=local_std,
        size=window_size,
        mode="constant",
        cval=np.nan
    )

    roughness[~water_mask] = np.nan

    return roughness


def compute_curvature(elevation, x, y, water_mask):
    """
    Compute terrain curvature as the absolute Laplacian
    of elevation using a five-point finite-difference stencil.

    Curvature is calculated only where the centre cell
    and its four axial neighbours are valid water cells.

    Returns
    -------
    curvature : 2D numpy array
        Absolute Laplacian of elevation, in 1/metres.
    """

    elevation = np.asarray(elevation, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)

    rows, cols = elevation.shape

    curvature = np.full((rows, cols), np.nan, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)

    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("Coordinates must have uniform spacing")

    dx = dx[0]
    dy = dy[0]

    center = elevation[1:-1, 1:-1]

    d2z_dx2 = (
        elevation[1:-1, 2:]
        - 2 * center
        + elevation[1:-1, :-2]
    ) / dx**2

    d2z_dy2 = (
        elevation[2:, 1:-1]
        - 2 * center
        + elevation[:-2, 1:-1]
    ) / dy**2

    valid = (
        water_mask[1:-1, 1:-1]
        & water_mask[1:-1, :-2]
        & water_mask[1:-1, 2:]
        & water_mask[:-2, 1:-1]
        & water_mask[2:, 1:-1]
    )

    valid &= (
        np.isfinite(center)
        & np.isfinite(elevation[1:-1, :-2])
        & np.isfinite(elevation[1:-1, 2:])
        & np.isfinite(elevation[:-2, 1:-1])
        & np.isfinite(elevation[2:, 1:-1])
    )

    laplacian = d2z_dx2 + d2z_dy2

    curvature[1:-1, 1:-1] = np.where(
        valid,
        np.abs(laplacian),
        np.nan
    )

    return curvature