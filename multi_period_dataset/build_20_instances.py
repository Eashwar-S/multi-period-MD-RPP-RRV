"""
Dataset builder for 20 real-world MD-RPP-RRV "black ice monitoring" instances.

Key changes vs your original script:
1) Uses Open‑Meteo *Historical Weather API* (hourly) instead of forecast daily aggregates.
2) Builds 20 instances from a config list (different US states + increasing graph size).
3) Avoids calling weather API for every node (too many calls). Instead samples K points
   (centroid + 4 corners of the graph bounding box) and assigns each node to nearest sample.
4) Writes a metadata CSV with computed node/edge counts, chosen time horizon, and paths.

Dependencies: osmnx, networkx, pandas, numpy, openmeteo-requests, requests-cache, retry-requests
"""

import os
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx

import openmeteo_requests
import requests_cache
from retry_requests import retry

# -----------------------------
# Open‑Meteo configuration
# -----------------------------
OPENMETEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"  # Historical Weather API
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "weather_code",
]

# Your planning loop is 30-min receding horizon; we collect hourly weather and
# (optionally) interpolate to 30-min later.
WEATHER_INTERVAL = "hourly"
PLANNING_INTERVAL_MIN = 30
SUNSET_WINDOW_LOCAL = ("16:00", "20:00")  # 4pm–8pm local refreeze window

# -----------------------------
# Graph configuration
# -----------------------------
ROADS_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'


@dataclass
class InstanceSpec:
    instance_id: int
    place: str
    state: str
    center_lat: float
    center_lon: float
    dist_m: int                 # controls graph size (increasing)
    intersection_tolerance_m: int
    timezone: str
    start_date: str             # YYYY-MM-DD
    end_date: str               # YYYY-MM-DD
    event_context: str


INSTANCE_SPECS: List[InstanceSpec] = [
    InstanceSpec(1, "Duluth", "MN", 46.7867, -92.1005, 3000, 80, "America/Chicago", "2026-01-23", "2026-01-26",
                 "Upper Midwest cold/ice potential (Jan 24–25, 2026 winter storm region)"),
    # InstanceSpec(2, "Fargo", "ND", 46.8772, -96.7898, 3500, 250, "America/Chicago", "2026-01-23", "2026-01-26",
    #              "Northern Plains cold/ice potential (Jan 24–26, 2026)"),
    # InstanceSpec(3, "Bismarck", "ND", 46.8083, -100.7837, 4000, 250, "America/Chicago", "2026-01-23", "2026-01-26",
    #              "Northern Plains cold/ice potential (Jan 24–26, 2026)"),
    # InstanceSpec(4, "Marquette", "MI", 46.5436, -87.3954, 4500, 250, "America/Detroit", "2026-01-23", "2026-01-26",
    #              "Great Lakes lake-enhanced snow/icing potential (Jan 24–25, 2026)"),
    # InstanceSpec(5, "Green Bay", "WI", 44.5133, -88.0133, 5000, 250, "America/Chicago", "2026-01-23", "2026-01-26",
    #              "Great Lakes snow/icing potential (Jan 24–25, 2026)"),
    # InstanceSpec(6, "Erie", "PA", 42.1292, -80.0851, 5500, 250, "America/New_York", "2025-11-25", "2025-11-30",
    #              "Great Lakes refreeze potential during late Nov 2025 lake-effect period"),
    # InstanceSpec(7, "Buffalo", "NY", 42.8864, -78.8784, 6000, 250, "America/New_York", "2025-11-26", "2025-11-30",
    #              "Lake-effect snow event (Nov 26–29, 2025)"),
    # InstanceSpec(8, "Rochester", "NY", 43.1566, -77.6088, 6500, 250, "America/New_York", "2025-11-26", "2025-11-30",
    #              "Lake-effect / post-snow refreeze potential (Nov 26–29, 2025)"),
    # InstanceSpec(9, "Syracuse", "NY", 43.0481, -76.1474, 7000, 250, "America/New_York", "2025-11-26", "2025-11-30",
    #              "Lake-effect / refreeze potential (Nov 26–29, 2025)"),
    # InstanceSpec(10, "Burlington", "VT", 44.4759, -73.2121, 7500, 250, "America/New_York", "2026-02-10", "2026-02-13",
    #              "New England clipper snow/ice (around Feb 11, 2026)"),
    # InstanceSpec(11, "Concord", "NH", 43.2081, -71.5376, 8000, 250, "America/New_York", "2026-02-10", "2026-02-13",
    #              "New England clipper snow/ice (around Feb 11, 2026)"),
    # InstanceSpec(12, "Portland", "ME", 43.6591, -70.2568, 8500, 250, "America/New_York", "2026-02-10", "2026-02-13",
    #              "New England clipper snow/ice (around Feb 11, 2026)"),
    # InstanceSpec(13, "Worcester", "MA", 42.2626, -71.8023, 9000, 250, "America/New_York", "2026-02-10", "2026-02-13",
    #              "New England clipper snow/ice (around Feb 11, 2026)"),
    # InstanceSpec(14, "Albany", "NY", 42.6526, -73.7562, 9500, 250, "America/New_York", "2026-02-10", "2026-02-13",
    #              "Interior Northeast clipper snow/ice (around Feb 11, 2026)"),
    # InstanceSpec(15, "Chicago", "IL", 41.8781, -87.6298, 10000, 250, "America/Chicago", "2025-11-09", "2025-11-12",
    #              "Lake-effect snow event (Nov 9–10, 2025)"),
    # InstanceSpec(16, "South Bend", "IN", 41.6764, -86.2520, 11000, 250, "America/Indiana/Indianapolis", "2025-11-09", "2025-11-12",
    #              "Lake-effect snow event (Nov 9–10, 2025)"),
    # InstanceSpec(17, "Salt Lake City", "UT", 40.7608, -111.8910, 12000, 250, "America/Denver", "2026-01-23", "2026-01-27",
    #              "Intermountain West winter conditions (Jan 24–27, 2026)"),
    # InstanceSpec(18, "Bozeman", "MT", 45.6770, -111.0429, 14000, 250, "America/Denver", "2026-01-23", "2026-01-27",
    #              "Northern Rockies winter conditions (Jan 24–27, 2026)"),
    # InstanceSpec(19, "Anchorage", "AK", 61.2181, -149.9003, 16000, 250, "America/Anchorage", "2025-12-10", "2025-12-14",
    #              "Alaska cold season (Dec 2025 stable cold window)"),
    # InstanceSpec(20, "Fairbanks", "AK", 64.8378, -147.7164, 18000, 250, "America/Anchorage", "2025-12-10", "2025-12-14",
    #              "Alaska cold season (Dec 2025 stable cold window)"),
]


def build_graph(spec: InstanceSpec) -> nx.MultiDiGraph:
    """
    Build a road network graph around a center point, then consolidate intersections.
    We use graph_from_point to control size consistently (dist_m increases across instances).
    """
    G = ox.graph_from_point(
        (spec.center_lat, spec.center_lon),
        dist=spec.dist_m,
        dist_type="bbox",
        retain_all=False,
        custom_filter=ROADS_FILTER,
        simplify=True,
    )
    # Project, consolidate, and project back to WGS84
    G_proj = ox.project_graph(G)
    G_cons = ox.consolidate_intersections(
        G_proj,
        tolerance=spec.intersection_tolerance_m,
        rebuild_graph=True,
        dead_ends=False,
    )
    G_wgs = ox.project_graph(G_cons, to_crs="EPSG:4326")
    return G_wgs


def sample_weather_points(G: nx.MultiDiGraph, k: int = 5) -> List[Tuple[float, float]]:
    """
    Sample a small set of points for weather queries:
    - centroid + 4 corners of graph bounding box (default k=5).
    """
    lats = np.array([data["y"] for _, data in G.nodes(data=True)])
    lons = np.array([data["x"] for _, data in G.nodes(data=True)])
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    centroid = (float(lats.mean()), float(lons.mean()))
    corners = [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_min),
        (lat_max, lon_max),
    ]
    points = [centroid] + corners
    return points[:k]


def openmeteo_client():
    cache_session = requests_cache.CachedSession(".cache_openmeteo", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_hourly_weather(points: List[Tuple[float, float]], start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """
    Query Open‑Meteo Historical Weather API for each sampled point and return one long dataframe.

    NOTE: This keeps API calls small and reproducible. Mapping from sampled points → nodes/edges
    can be done later using nearest-point assignment.
    """
    client = openmeteo_client()
    all_rows = []

    for i, (lat, lon) in enumerate(points):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": HOURLY_VARS,
            "timezone": timezone,
        }
        responses = client.weather_api(OPENMETEO_HISTORICAL_URL, params=params)
        r = responses[0]
        hourly = r.Hourly()

        # Time grid
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(hours=1),
            inclusive="left",
        )

        data = {"sample_id": i, "latitude": lat, "longitude": lon, "time_utc": times}

        # Extract each hourly variable by index (same order as HOURLY_VARS)
        for j, var in enumerate(HOURLY_VARS):
            data[var] = hourly.Variables(j).ValuesAsNumpy()

        all_rows.append(pd.DataFrame(data))

    return pd.concat(all_rows, ignore_index=True)


def assign_nodes_to_samples(G: nx.MultiDiGraph, sample_points: List[Tuple[float, float]]) -> Dict[int, int]:
    """
    Assign each node to nearest weather sample point (by lat/lon Euclidean distance).
    Returns mapping: node_id -> sample_id
    """
    pts = np.array(sample_points)  # (lat, lon)
    node_ids = []
    node_xy = []
    for nid, data in G.nodes(data=True):
        node_ids.append(nid)
        node_xy.append((data["y"], data["x"]))
    node_xy = np.array(node_xy)

    # Compute nearest sample point for each node
    # (vectorized; OK for a few 10^4 nodes)
    d2 = ((node_xy[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    return {nid: int(sid) for nid, sid in zip(node_ids, nearest)}


def write_instance(spec: InstanceSpec, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    graphs_dir = os.path.join(out_dir, "graphs")
    weather_dir = os.path.join(out_dir, "weather")
    meta_dir = os.path.join(out_dir, "meta")
    for d in (graphs_dir, weather_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    # 1) Graph
    G = build_graph(spec)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    graph_path = os.path.join(graphs_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}.pickle")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    # 2) Weather sample points
    sample_pts = sample_weather_points(G, k=5)
    weather_df = fetch_hourly_weather(sample_pts, spec.start_date, spec.end_date, spec.timezone)
    weather_path = os.path.join(weather_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_hourly.csv")
    weather_df.to_csv(weather_path, index=False)

    # 3) Node → sample assignment
    node_to_sample = assign_nodes_to_samples(G, sample_pts)
    mapping_path = os.path.join(meta_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_node_to_sample.json")
    with open(mapping_path, "w") as f:
        json.dump(node_to_sample, f)

    # 4) Metadata
    meta = asdict(spec)
    meta.update({
        "computed_nodes": int(n_nodes),
        "computed_edges": int(n_edges),
        "weather_interval": WEATHER_INTERVAL,
        "planning_interval_min": PLANNING_INTERVAL_MIN,
        "sunset_window_local": {"start": SUNSET_WINDOW_LOCAL[0], "end": SUNSET_WINDOW_LOCAL[1]},
        "graph_path": graph_path,
        "weather_path": weather_path,
        "node_to_sample_path": mapping_path,
        "n_weather_samples": len(sample_pts),
    })
    meta_path = os.path.join(meta_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def build_all_instances(out_dir: str = "instances_out") -> pd.DataFrame:
    rows = []
    for spec in INSTANCE_SPECS:
        print(f"\nBuilding instance {spec.instance_id:02d}: {spec.place}, {spec.state} (dist={spec.dist_m}m)")
        meta = write_instance(spec, out_dir=out_dir)
        rows.append(meta)

    df = pd.DataFrame(rows)
    # Compact metadata CSV (easy to paste into dissertation tables)
    df_out = df[[
        "instance_id","place","state",
        "computed_nodes","computed_edges",
        "dist_m","intersection_tolerance_m",
        "timezone","start_date","end_date",
        "weather_interval","planning_interval_min",
        "n_weather_samples","event_context",
        "graph_path","weather_path","node_to_sample_path"
    ]]
    csv_path = os.path.join(out_dir, "instances_metadata.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")
    return df_out


if __name__ == "__main__":
    # Run once to generate all 20 instances and metadata.
    build_all_instances(out_dir="md_rpp_rrv_ice_instances")
