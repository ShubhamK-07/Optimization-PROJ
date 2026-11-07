"""GEBCO loading and metric-grid preprocessing for the Gibraltar tunnel project.

Run from the repository root:
    python -m src.data --config config.yaml

The pipeline keeps the raw GEBCO file unchanged and writes aligned processed
arrays plus metadata under data/processed/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from pyproj import Geod, Transformer


def load_gebco(path: str | Path, bbox: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an elevation subset from a GEBCO NetCDF file.

    Returns longitude, latitude, elevation (metres; negative below sea level).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GEBCO NetCDF not found: {path}")

    with xr.open_dataset(path) as ds:
        if "elevation" not in ds:
            raise KeyError("Expected an 'elevation' variable in the GEBCO NetCDF.")
        sub = ds.sel(
            lon=slice(bbox["west"], bbox["east"]),
            lat=slice(bbox["south"], bbox["north"]),
        )
        lon = np.asarray(sub["lon"].values, dtype=float)
        lat = np.asarray(sub["lat"].values, dtype=float)
        elevation = np.asarray(sub["elevation"].values, dtype=np.float32)

    if elevation.shape != (lat.size, lon.size):
        raise ValueError("Unexpected GEBCO array shape; expected (latitude, longitude).")
    if lon.size < 2 or lat.size < 2:
        raise ValueError("The selected bounding box contains too few grid cells.")
    return lon, lat, elevation


def project_coordinates(
    lon: np.ndarray, lat: np.ndarray, epsg: str
) -> tuple[np.ndarray, np.ndarray, Transformer, Transformer]:
    """Create forward and inverse geographic/projected coordinate transforms."""
    forward = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    inverse = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = forward.transform(lon2d, lat2d)
    return np.asarray(x), np.asarray(y), forward, inverse


def resample_bathymetry(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    inverse: Transformer,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample onto a regular projected grid using nearest source-cell sampling.

    Nearest-neighbour sampling preserves original GEBCO cell values and avoids
    interpolating across the land/water boundary. The output extent is based on
    the projected outer edges of the source grid.
    """
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive.")

    dlon = float(np.median(np.diff(lon)))
    dlat = float(np.median(np.diff(lat)))
    west, east = float(lon.min() - dlon / 2), float(lon.max() + dlon / 2)
    south, north = float(lat.min() - dlat / 2), float(lat.max() + dlat / 2)

    edge_lon = np.concatenate([
        np.linspace(west, east, 200),
        np.full(200, west),
        np.full(200, east),
    ])
    edge_lat = np.concatenate([
        np.full(200, south),
        np.linspace(south, north, 200),
        np.linspace(south, north, 200),
    ])
    geographic_to_projected = Transformer.from_crs(
        "EPSG:4326", inverse.source_crs, always_xy=True
    )
    edge_x, edge_y = geographic_to_projected.transform(edge_lon, edge_lat)

    x = np.arange(np.min(edge_x) + resolution_m / 2, np.max(edge_x), resolution_m)
    y = np.arange(np.min(edge_y) + resolution_m / 2, np.max(edge_y), resolution_m)
    xx, yy = np.meshgrid(x, y)
    target_lon, target_lat = inverse.transform(xx, yy)

    lon_idx = np.abs(lon[None, None, :] - target_lon[:, :, None]).argmin(axis=2)
    lat_idx = np.abs(lat[None, None, :] - target_lat[:, :, None]).argmin(axis=2)
    sampled = elevation[lat_idx, lon_idx].copy()

    inside = (
        (target_lon >= west) & (target_lon <= east)
        & (target_lat >= south) & (target_lat <= north)
    )
    sampled[~inside] = np.nan
    return x, y, sampled.astype(np.float32)


def validate_distance(
    endpoints: dict, forward: Transformer
) -> dict:
    """Compare projected endpoint distance with WGS84 geodesic distance."""
    a_lon, a_lat = endpoints["A"]["lon"], endpoints["A"]["lat"]
    b_lon, b_lat = endpoints["B"]["lon"], endpoints["B"]["lat"]
    ax, ay = forward.transform(a_lon, a_lat)
    bx, by = forward.transform(b_lon, b_lat)
    projected = float(np.hypot(bx - ax, by - ay))
    geod = Geod(ellps="WGS84")
    _, _, geodesic = geod.inv(a_lon, a_lat, b_lon, b_lat)
    return {
        "A_projected_m": {"x": float(ax), "y": float(ay)},
        "B_projected_m": {"x": float(bx), "y": float(by)},
        "projected_distance_m": projected,
        "geodesic_distance_m": float(geodesic),
        "absolute_difference_m": float(abs(projected - geodesic)),
        "relative_difference_percent": float(abs(projected - geodesic) / geodesic * 100),
    }


def run_pipeline(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = root / config["data"]["raw_file"]
    lon, lat, elevation = load_gebco(raw_path, config["region"]["bbox"])
    _, _, forward, inverse = project_coordinates(
        lon, lat, config["projection"]["epsg"]
    )
    x, y, z = resample_bathymetry(
        lon, lat, elevation, inverse, float(config["projection"]["resolution_m"])
    )

    # Positive-down water depth. Land and outside-subset cells remain masked.
    water_mask = np.isfinite(z) & (z < 0)
    depth = np.where(water_mask, -z, np.nan).astype(np.float32)

    # Locate endpoints on nearest projected grid cells and validate the sampled cell.
    endpoint_info = {}
    for label in ("A", "B"):
        lon_e = float(config["endpoints"][label]["lon"])
        lat_e = float(config["endpoints"][label]["lat"])
        ex, ey = forward.transform(lon_e, lat_e)
        ix = int(np.abs(x - ex).argmin())
        iy = int(np.abs(y - ey).argmin())
        endpoint_info[label] = {
            "requested_lon": lon_e, "requested_lat": lat_e,
            "grid_x_m": float(x[ix]), "grid_y_m": float(y[iy]),
            "column": ix, "row": iy,
            "sampled_elevation_m": float(z[iy, ix]),
            "sampled_depth_m": float(depth[iy, ix]) if water_mask[iy, ix] else None,
            "is_water": bool(water_mask[iy, ix]),
        }

    distance_check = validate_distance(config["endpoints"], forward)

    processed_dir = root / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    npz_path = processed_dir / "gibraltar_metric_grid.npz"
    np.savez_compressed(
        npz_path, x=x, y=y, elevation=z, depth=depth, water_mask=water_mask
    )

    valid = np.isfinite(z)
    metadata = {
        "dataset": "GEBCO_2026",
        "source_file": str(Path(config["data"]["raw_file"])),
        "bbox": config["region"]["bbox"],
        "crs": config["projection"]["epsg"],
        "resolution_m": float(config["projection"]["resolution_m"]),
        "resampling": "nearest-neighbour source-cell sampling",
        "shape_rows_columns": [int(z.shape[0]), int(z.shape[1])],
        "valid_cells": int(valid.sum()),
        "total_cells": int(z.size),
        "land_cells": int(np.sum(valid & (z >= 0))),
        "water_cells": int(water_mask.sum()),
        "land_fraction_of_valid_cells": float(np.sum(valid & (z >= 0)) / valid.sum()),
        "elevation_min_m": float(np.nanmin(z)),
        "elevation_max_m": float(np.nanmax(z)),
        "depth_min_m": float(np.nanmin(depth)),
        "depth_max_m": float(np.nanmax(depth)),
        "endpoints": endpoint_info,
        "distance_validation": distance_check,
        "notes": [
            "Exploratory working crop; final study bounds may be refined.",
            "Depth is positive downward and represents seafloor depth, not tunnel depth below seabed.",
            "Land is excluded from water_mask; original elevation values are retained in the array.",
        ],
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"npz": npz_path, "metadata": metadata_path, "summary": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the Gibraltar GEBCO subset.")
    parser.add_argument("--config", default="config.yaml", help="Path to project config YAML.")
    args = parser.parse_args()
    result = run_pipeline(args.config)
    print(f"Processed arrays: {result['npz']}")
    print(f"Metadata:         {result['metadata']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
