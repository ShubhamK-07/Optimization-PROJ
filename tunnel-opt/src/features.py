import numpy as np

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