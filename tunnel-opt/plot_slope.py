import numpy as np
import matplotlib.pyplot as plt

from src.features import compute_slope

data = np.load("data/processed/gibraltar_metric_grid.npz")

slope = compute_slope(
    data["elevation"],
    data["x"],
    data["y"],
    data["water_mask"]
)

plt.figure(figsize=(12, 6))

plt.imshow(
    slope,
    origin="lower",
    extent=[
        data["x"].min(),
        data["x"].max(),
        data["y"].min(),
        data["y"].max()
    ],
    cmap="viridis",
    vmin=0,
    vmax=0.5
)

plt.colorbar(label="Slope (m/m)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Seafloor Slope — Strait of Gibraltar")

plt.tight_layout()
plt.show()