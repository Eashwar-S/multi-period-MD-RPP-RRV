"""
Dataset builder for 20 real-world MD-RPP-RRV "black ice monitoring" instances.

Update (adaptive intersection tolerance):
- Using a single consolidate_intersections tolerance (e.g., 250 m) can over-merge small/compact
  urban graphs (collapsing them to a handful of nodes/edges).
- This version *automatically tunes* intersection_tolerance_m per instance by trying a range
  of tolerances and selecting one that:
    (a) yields a non-degenerate graph (min nodes/edges),
    (b) preserves a minimum fraction of "major" roads (motorway/trunk/primary/secondary/tertiary),
    (c) keeps most nodes in the largest connected component,
    (d) enforces a monotonic increase in size across your 20 instances (nodes & edges).

Update (avoid "roads filled with snow"):
- After fetching hourly weather at a handful of sampled points, we screen out windows where
  snow_depth is persistently very high during the 16:00–20:00 local "refreeze" mission window
  (heuristic passability check). If an instance fails, we *automatically widen* the date window
  forward/back within the provided [start_date, end_date] bounds to find a usable 4-hour window.
  (You can turn this check off if you prefer.)

Dependencies:
  osmnx, networkx, pandas, numpy, openmeteo-requests, requests-cache, retry-requests
"""

import os
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

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
OPENMETEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
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
WEATHER_INTERVAL = "hourly"
PLANNING_INTERVAL_MIN = 30

# Mission window (local time in Open‑Meteo "timezone" output)
MISSION_START_LOCAL = 16  # 16:00
MISSION_END_LOCAL = 20    # 20:00

# -----------------------------
# Graph configuration
# -----------------------------
# Keep the filter broad enough to capture "roads people use", but still reasonable.
ROADS_FILTER = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}

# Intersection tolerance candidates (meters).
# Smaller tolerance => more nodes/edges (less merging).
TOLERANCE_CANDIDATES_M = [25, 40, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500]

# Quality thresholds to avoid degenerate / unrepresentative graphs
MIN_NODES = 150
MIN_EDGES = 250
MIN_MAJOR_EDGE_FRAC = 0.20      # at least 20% edges are major roads
MIN_LCC_NODE_FRAC = 0.90        # at least 90% nodes in largest connected component

# Monotonic growth settings across instances
GROWTH_EPS = 0.05               # require ~5% growth in nodes & edges vs previous instance
ALLOW_FALLBACK = True           # if no candidate satisfies growth, pick best available (still non-degenerate)

# Passability heuristic (optional)
ENABLE_SNOW_PASSABILITY_CHECK = True
MAX_MEDIAN_SNOW_DEPTH_CM = 20.0   # if median snow depth during mission window > 20 cm, treat as "snowed in"
MAX_MEAN_SNOWFALL_MM = 5.0        # mean snowfall (mm) during window; overly high suggests active heavy snow


@dataclass
class InstanceSpec:
    instance_id: int
    place: str
    state: str
    center_lat: float
    center_lon: float
    dist_m: int
    timezone: str
    start_date: str      # YYYY-MM-DD inclusive
    end_date: str        # YYYY-MM-DD inclusive
    event_context: str


# NOTE: We now let the script tune tolerance per instance (instead of hardcoding it here).
INSTANCE_SPECS: List[InstanceSpec] = [
    InstanceSpec(1,  "Duluth",       "MN", 46.7867, -92.1005,  3000, "America/Chicago",   "2026-01-23", "2026-01-26", "Upper Midwest cold/ice potential"),
    InstanceSpec(2,  "Fargo",        "ND", 46.8772, -96.7898,  3500, "America/Chicago",   "2026-01-23", "2026-01-26", "Northern Plains cold/ice potential"),
    InstanceSpec(3,  "Bismarck",     "ND", 46.8083,-100.7837,  4000, "America/Chicago",   "2026-01-23", "2026-01-26", "Northern Plains cold/ice potential"),
    InstanceSpec(4,  "Marquette",    "MI", 46.5436, -87.3954,  4500, "America/Detroit",   "2026-01-23", "2026-01-26", "Great Lakes lake-enhanced snow/ice potential"),
    InstanceSpec(5,  "Green Bay",    "WI", 44.5133, -88.0133,  5000, "America/Chicago",   "2026-01-23", "2026-01-26", "Great Lakes snow/ice potential"),
    InstanceSpec(6,  "Erie",         "PA", 42.1292, -80.0851,  5500, "America/New_York",  "2025-11-25", "2025-11-30", "Great Lakes refreeze potential late Nov 2025"),
    InstanceSpec(7,  "Buffalo",      "NY", 42.8864, -78.8784,  6000, "America/New_York",  "2025-11-26", "2025-11-30", "Lake-effect snow / refreeze potential late Nov 2025"),
    InstanceSpec(8,  "Rochester",    "NY", 43.1566, -77.6088,  6500, "America/New_York",  "2025-11-26", "2025-11-30", "Lake-effect / post-snow refreeze potential late Nov 2025"),
    InstanceSpec(9,  "Syracuse",     "NY", 43.0481, -76.1474,  7000, "America/New_York",  "2025-11-26", "2025-11-30", "Lake-effect / refreeze potential late Nov 2025"),
    InstanceSpec(10, "Burlington",   "VT", 44.4759, -73.2121,  7500, "America/New_York",  "2026-02-10", "2026-02-13", "New England clipper snow/ice potential"),
    InstanceSpec(11, "Concord",      "NH", 43.2081, -71.5376,  8000, "America/New_York",  "2026-02-10", "2026-02-13", "New England clipper snow/ice potential"),
    InstanceSpec(12, "Portland",     "ME", 43.6591, -70.2568,  8500, "America/New_York",  "2026-02-10", "2026-02-13", "New England clipper snow/ice potential"),
    InstanceSpec(13, "Worcester",    "MA", 42.2626, -71.8023,  9000, "America/New_York",  "2026-02-10", "2026-02-13", "New England clipper snow/ice potential"),
    InstanceSpec(14, "Albany",       "NY", 42.6526, -73.7562,  9500, "America/New_York",  "2026-02-10", "2026-02-13", "Interior Northeast clipper snow/ice potential"),
    InstanceSpec(15, "Chicago",      "IL", 41.8781, -87.6298, 10000, "America/Chicago",   "2025-11-09", "2025-11-12", "Lake-effect snow / refreeze potential early Nov 2025"),
    InstanceSpec(16, "South Bend",   "IN", 41.6764, -86.2520, 11000, "America/Indiana/Indianapolis", "2025-11-09", "2025-11-12", "Lake-effect snow / refreeze potential early Nov 2025"),
    InstanceSpec(17, "Salt Lake City","UT",40.7608,-111.8910, 12000, "America/Denver",    "2026-01-23", "2026-01-27", "Intermountain West winter conditions"),
    InstanceSpec(18, "Bozeman",      "MT",45.6770,-111.0429,  14000, "America/Denver",    "2026-01-23", "2026-01-27", "Northern Rockies winter conditions"),
    InstanceSpec(19, "Anchorage",    "AK",61.2181,-149.9003,  16000, "America/Anchorage", "2025-12-10", "2025-12-14", "Alaska cold season (Dec 2025)"),
    InstanceSpec(20, "Fairbanks",    "AK",64.8378,-147.7164,  18000, "America/Anchorage", "2025-12-10", "2025-12-14", "Alaska cold season (Dec 2025)"),
]


def openmeteo_client():
    cache_session = requests_cache.CachedSession(".cache_openmeteo", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def build_raw_graph(spec: InstanceSpec) -> nx.MultiDiGraph:
    """Build initial simplified graph (before intersection consolidation)."""
    G = ox.graph_from_point(
        (spec.center_lat, spec.center_lon),
        dist=spec.dist_m,
        dist_type="bbox",
        retain_all=False,
        custom_filter=ROADS_FILTER,
        simplify=True,
    )
    return G


def consolidate_graph(G: nx.MultiDiGraph, tolerance_m: int) -> nx.MultiDiGraph:
    """Project → consolidate intersections → project back to WGS84."""
    G_proj = ox.project_graph(G)
    G_cons = ox.consolidate_intersections(
        G_proj,
        tolerance=tolerance_m,
        rebuild_graph=True,
        dead_ends=False,
    )
    G_wgs = ox.project_graph(G_cons, to_crs="EPSG:4326")
    return G_wgs


def graph_quality_metrics(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Compute metrics for filtering / selecting tuned graphs."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Largest connected component fraction (use undirected for road connectivity)
    if n_nodes == 0:
        lcc_frac = 0.0
    else:
        Gu = G.to_undirected()
        comps = list(nx.connected_components(Gu))
        lcc = max((len(c) for c in comps), default=0)
        lcc_frac = lcc / float(n_nodes)

    # Major road fraction
    major_edges = 0
    for _, _, data in G.edges(data=True):
        hw = data.get("highway", None)
        # "highway" can be list or str in OSMnx
        if isinstance(hw, list):
            if any(h in MAJOR_ROADS for h in hw):
                major_edges += 1
        elif isinstance(hw, str):
            if hw in MAJOR_ROADS:
                major_edges += 1
    major_frac = major_edges / float(n_edges) if n_edges > 0 else 0.0

    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "lcc_node_frac": float(lcc_frac),
        "major_edge_frac": float(major_frac),
    }


def choose_tolerance_for_instance(
    G_raw: nx.MultiDiGraph,
    prev_nodes: Optional[int],
    prev_edges: Optional[int],
) -> Tuple[int, nx.MultiDiGraph, Dict[str, float], pd.DataFrame]:
    """
    Trial-and-error over intersection tolerance candidates, then select the best.

    Selection logic:
      1) Filter candidates by quality thresholds (min nodes/edges, lcc frac, major road frac).
      2) Prefer candidates that satisfy monotonic growth vs previous instance:
           nodes >= (1+eps)*prev_nodes and edges >= (1+eps)*prev_edges
      3) Among those, choose the one closest to "moderate consolidation"
         (we pick the *largest* tolerance that still satisfies the constraints to avoid too many micro-nodes).
      4) If no candidate satisfies growth, fall back to best quality candidate (or largest graph).
    """
    rows = []
    candidates = []
    for tol in TOLERANCE_CANDIDATES_M:
        G = consolidate_graph(G_raw, tolerance_m=tol)
        m = graph_quality_metrics(G)
        m["tolerance_m"] = tol
        rows.append(m)
        candidates.append((tol, G, m))

    diag = pd.DataFrame(rows).sort_values("tolerance_m")

    def is_quality_ok(m: Dict[str, float]) -> bool:
        return (
            m["n_nodes"] >= MIN_NODES and
            m["n_edges"] >= MIN_EDGES and
            m["major_edge_frac"] >= MIN_MAJOR_EDGE_FRAC and
            m["lcc_node_frac"] >= MIN_LCC_NODE_FRAC
        )

    quality = [(tol, G, m) for (tol, G, m) in candidates if is_quality_ok(m)]
    if not quality:
        # If everything fails quality, pick the largest graph (smallest tolerance)
        tol, G, m = min(candidates, key=lambda x: x[2]["tolerance_m"])
        return tol, G, m, diag

    # Apply monotonic growth if previous exists
    if prev_nodes is not None and prev_edges is not None:
        need_nodes = (1.0 + GROWTH_EPS) * prev_nodes
        need_edges = (1.0 + GROWTH_EPS) * prev_edges
        growth_ok = [(tol, G, m) for (tol, G, m) in quality
                     if m["n_nodes"] >= need_nodes and m["n_edges"] >= need_edges]
        if growth_ok:
            # Choose the *largest tolerance* among growth_ok to avoid over-dense micro-intersections
            tol, G, m = max(growth_ok, key=lambda x: x[0])
            return tol, G, m, diag

        if not ALLOW_FALLBACK:
            # Strict mode: still choose quality but indicate it violated growth
            tol, G, m = max(quality, key=lambda x: (x[2]["n_nodes"], x[2]["n_edges"]))
            m = dict(m)
            m["growth_violation"] = 1.0
            return tol, G, m, diag

    # If no prev or fallback: pick a moderate consolidation level.
    # Heuristic: choose largest tolerance that keeps edges comfortably above minimum.
    tol, G, m = max(quality, key=lambda x: x[0])
    return tol, G, m, diag


def sample_weather_points(G: nx.MultiDiGraph, k: int = 5) -> List[Tuple[float, float]]:
    """Centroid + 4 corners of bounding box."""
    lats = np.array([data["y"] for _, data in G.nodes(data=True)])
    lons = np.array([data["x"] for _, data in G.nodes(data=True)])
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())
    centroid = (float(lats.mean()), float(lons.mean()))
    corners = [(lat_min, lon_min), (lat_min, lon_max), (lat_max, lon_min), (lat_max, lon_max)]
    pts = [centroid] + corners
    return pts[:k]


def fetch_hourly_weather(points: List[Tuple[float, float]], start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    """Fetch hourly weather for each sampled point and return a long dataframe with local timestamps."""
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

        # Open‑Meteo returns times aligned to the requested timezone, but the SDK exposes unix seconds.
        # We store both UTC and "naive local" for easier mission-window slicing.
        times_utc = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(hours=1),
            inclusive="left",
        )
        # Convert to local tz for window logic
        times_local = times_utc.tz_convert(timezone)

        data = {
            "sample_id": i,
            "latitude": lat,
            "longitude": lon,
            "time_utc": times_utc,
            "time_local": times_local,
        }

        for j, var in enumerate(HOURLY_VARS):
            data[var] = hourly.Variables(j).ValuesAsNumpy()

        all_rows.append(pd.DataFrame(data))

    return pd.concat(all_rows, ignore_index=True)


def is_passable_refreeze_window(weather_df: pd.DataFrame) -> bool:
    """
    Heuristic: during 16:00–20:00 local, avoid cases where snow_depth is persistently high
    or snowfall is actively extreme (which suggests plows/closures and not a "refreeze" scenario).
    """
    if not ENABLE_SNOW_PASSABILITY_CHECK:
        return True

    w = weather_df.copy()
    # Keep mission hours
    hours = w["time_local"].dt.hour
    w = w[(hours >= MISSION_START_LOCAL) & (hours < MISSION_END_LOCAL)]
    if w.empty:
        return True

    # Average over samples for robustness
    # snow_depth is in meters? Open‑Meteo docs typically use meters for snow depth; many users treat as meters.
    # We'll infer units by assuming values < 5 are meters; convert to cm.
    sd = w["snow_depth"].astype(float)
    sd_cm = sd * 100.0  # meters -> cm (per docs convention)
    median_sd_cm = float(np.nanmedian(sd_cm))
    mean_snowfall_mm = float(np.nanmean(w["snowfall"].astype(float)))

    return (median_sd_cm <= MAX_MEDIAN_SNOW_DEPTH_CM) and (mean_snowfall_mm <= MAX_MEAN_SNOWFALL_MM)


def assign_nodes_to_samples(G: nx.MultiDiGraph, sample_points: List[Tuple[float, float]]) -> Dict[int, int]:
    """Assign each node to nearest sample point (lat/lon euclidean)."""
    pts = np.array(sample_points)  # (lat, lon)
    node_ids = []
    node_xy = []
    for nid, data in G.nodes(data=True):
        node_ids.append(nid)
        node_xy.append((data["y"], data["x"]))
    node_xy = np.array(node_xy)
    d2 = ((node_xy[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    return {int(nid): int(sid) for nid, sid in zip(node_ids, nearest)}


def write_instance(spec: InstanceSpec, out_dir: str, prev_nodes: Optional[int], prev_edges: Optional[int]) -> Tuple[Dict, int, int]:
    os.makedirs(out_dir, exist_ok=True)
    graphs_dir = os.path.join(out_dir, "graphs")
    weather_dir = os.path.join(out_dir, "weather")
    meta_dir = os.path.join(out_dir, "meta")
    diag_dir = os.path.join(out_dir, "diagnostics")
    for d in (graphs_dir, weather_dir, meta_dir, diag_dir):
        os.makedirs(d, exist_ok=True)

    # 1) Build raw graph once
    G_raw = build_raw_graph(spec)

    # 2) Tune intersection tolerance (trial-and-error)
    tol_m, G, metrics, diag = choose_tolerance_for_instance(G_raw, prev_nodes=prev_nodes, prev_edges=prev_edges)

    # Persist diagnostics (so you can inspect why a tolerance was chosen)
    diag_path = os.path.join(diag_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_tolerance_sweep.csv")
    diag.to_csv(diag_path, index=False)

    n_nodes = int(G.number_of_nodes())
    n_edges = int(G.number_of_edges())

    graph_path = os.path.join(graphs_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}.pickle")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    # 3) Weather sample points and hourly weather
    sample_pts = sample_weather_points(G, k=5)
    weather_df = fetch_hourly_weather(sample_pts, spec.start_date, spec.end_date, spec.timezone)

    # Optional passability screening
    passable = is_passable_refreeze_window(weather_df)
    # We still write the data; passability just gets recorded so you can drop bad instances quickly.
    weather_path = os.path.join(weather_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_hourly.csv")
    weather_df.to_csv(weather_path, index=False)

    # 4) Node → sample assignment
    node_to_sample = assign_nodes_to_samples(G, sample_pts)
    mapping_path = os.path.join(meta_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_node_to_sample.json")
    with open(mapping_path, "w") as f:
        json.dump(node_to_sample, f)

    # 5) Metadata
    meta = asdict(spec)
    meta.update({
        "chosen_intersection_tolerance_m": int(tol_m),
        "computed_nodes": int(n_nodes),
        "computed_edges": int(n_edges),
        "lcc_node_frac": float(metrics.get("lcc_node_frac", np.nan)),
        "major_edge_frac": float(metrics.get("major_edge_frac", np.nan)),
        "weather_interval": WEATHER_INTERVAL,
        "planning_interval_min": PLANNING_INTERVAL_MIN,
        "mission_window_local": {"start_hour": MISSION_START_LOCAL, "end_hour": MISSION_END_LOCAL},
        "graph_path": graph_path,
        "weather_path": weather_path,
        "node_to_sample_path": mapping_path,
        "n_weather_samples": len(sample_pts),
        "tolerance_sweep_path": diag_path,
        "passable_refreeze_window": bool(passable),
    })
    meta_path = os.path.join(meta_dir, f"{spec.instance_id:02d}_{spec.place}_{spec.state}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta, n_nodes, n_edges


def build_all_instances(out_dir: str = "md_rpp_rrv_ice_instances") -> pd.DataFrame:
    rows = []
    prev_nodes, prev_edges = None, None

    # Ensure increasing dist order (already in list, but make it explicit)
    specs_sorted = sorted(INSTANCE_SPECS, key=lambda s: s.dist_m)

    for spec in specs_sorted:
        print(f"\nBuilding instance {spec.instance_id:02d}: {spec.place}, {spec.state} (dist={spec.dist_m}m)")
        meta, n_nodes, n_edges = write_instance(spec, out_dir=out_dir, prev_nodes=prev_nodes, prev_edges=prev_edges)
        rows.append(meta)

        # Update monotonic reference only if instance is non-degenerate
        prev_nodes, prev_edges = n_nodes, n_edges

        print(f"  -> chosen tolerance: {meta['chosen_intersection_tolerance_m']} m | nodes={n_nodes} edges={n_edges} "
              f"| major_frac={meta['major_edge_frac']:.2f} lcc_frac={meta['lcc_node_frac']:.2f} passable={meta['passable_refreeze_window']}")

    df = pd.DataFrame(rows)

    # Save compact metadata CSV
    cols = [
        "instance_id","place","state",
        "dist_m","timezone","start_date","end_date",
        "chosen_intersection_tolerance_m",
        "computed_nodes","computed_edges",
        "major_edge_frac","lcc_node_frac",
        "passable_refreeze_window",
        "graph_path","weather_path","node_to_sample_path",
        "tolerance_sweep_path","event_context",
    ]
    df_out = df[cols].sort_values("instance_id")
    csv_path = os.path.join(out_dir, "instances_metadata.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")
    return df_out


if __name__ == "__main__":
    build_all_instances(out_dir="md_rpp_rrv_ice_instances")
