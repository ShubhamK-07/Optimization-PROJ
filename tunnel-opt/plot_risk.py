import numpy as np
import matplotlib.pyplot as plt

from src.features import (
    compute_slope,
    compute_roughness,
    compute_curvature,
    compute_risk,
)

data = np.load("data/processed/gibraltar_metric_grid.npz")

x = data["x"]
y = data["y"]
elevation = data["elevation"]
water_mask = data["water_mask"]

slope = compute_slope(elevation, x, y, water_mask)
roughness = compute_roughness(slope, water_mask)
curvature = compute_curvature(elevation, x, y, water_mask)

risk = compute_risk(slope, roughness, curvature)

plt.figure(figsize=(12, 6))

plt.imshow(
    risk,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="inferno",
)

plt.colorbar(label="Terrain Risk (0–1)")
plt.title("Combined Terrain Risk Field")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.tight_layout()
plt.show()