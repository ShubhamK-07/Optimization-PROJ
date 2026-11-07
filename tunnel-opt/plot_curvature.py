import numpy as np
import matplotlib.pyplot as plt

from src.features import compute_curvature

data = np.load("data/processed/gibraltar_metric_grid.npz")

curvature = compute_curvature(
    data["elevation"],
    data["x"],
    data["y"],
    data["water_mask"]
)

plt.figure(figsize=(12, 6))

plt.imshow(
    curvature,
    origin="lower",
    extent=[
        data["x"].min(),
        data["x"].max(),
        data["y"].min(),
        data["y"].max()
    ],
    cmap="inferno",
    vmin=0,
    vmax=0.0056
)

plt.colorbar(label="Absolute Laplacian (1/m)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Seafloor Terrain Curvature — Strait of Gibraltar")

plt.tight_layout()
plt.show()