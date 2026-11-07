import numpy as np
import matplotlib.pyplot as plt

from src.features import compute_slope, compute_roughness

data = np.load("data/processed/gibraltar_metric_grid.npz")

slope = compute_slope(
    data["elevation"],
    data["x"],
    data["y"],
    data["water_mask"]
)

roughness = compute_roughness(
    slope,
    data["water_mask"]
)

plt.figure(figsize=(12, 6))

plt.imshow(
    roughness,
    origin="lower",
    extent=[
        data["x"].min(),
        data["x"].max(),
        data["y"].min(),
        data["y"].max()
    ],
    cmap="magma",
    vmin=0,
    vmax=0.16
)

plt.colorbar(label="Terrain roughness (slope std. dev.)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Seafloor Terrain Roughness — Strait of Gibraltar")

plt.tight_layout()
plt.show()