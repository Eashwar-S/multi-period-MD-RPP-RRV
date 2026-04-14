"""
build_real_world_graphs.py
--------------------------
Downloads, consolidates, and saves the 13 real-world road-network graphs
defined in multi_period_dataset/Final_dataset_info.xlsx.

Graphs are saved as .pickle files inside:
    multi_period_dataset/real_world_instances/

Each graph is also visualised with lat (x-axis) and long (y-axis) and
the figure is saved alongside the pickle as a .png.
"""

import os
import re
import pickle
import warnings

import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

warnings.filterwarnings("ignore")
matplotlib.use("Agg")          # non-interactive backend - safe for scripts

# -- Paths -----------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH   = os.path.join(SCRIPT_DIR, "multi_period_dataset", "Final_dataset_info.xlsx")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "multi_period_dataset", "real_world_instances")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Hardcoded instance coordinates ----------------------------------------------
# Keys match the SNO column in Final_dataset_info.xlsx (1-indexed strings).
# Update coords here to fix any mismatches with the recorded dataset values.
INSTANCE_COORDS = {
    '1':  {'north': 39.00164, 'east': -76.90541, 'south': 38.91214, 'west': -77.02, 'tolerance': 56},  # College Park, MD
    '2':  {'north': 32.77486, 'east': -97.04704, 'south': 32.67753, 'west': -97.17201, 'tolerance': 180},  # Arlington, TX
    '3':  {'north': 33.62610, 'east': -101.79180,'south': 33.52370, 'west': -101.94350,'tolerance': 160},  # Lubbock, TX
    '4':  {'north': 39.99619, 'east': -75.12794, 'south': 39.89460, 'west': -75.29016, 'tolerance': 130},  # Philadelphia, PA
    '5':  {'north': 35.52, 'east': -97.47, 'south': 35.42, 'west': -97.58, 'tolerance': 135},  # Oklahoma City, OK
    '6':  {'north': 34.80, 'east': -92.24, 'south': 34.70, 'west': -92.34, 'tolerance': 125},  # Little Rock, AR  (East sign corrected)
    '7':  {'north': 35.20, 'east': -90.00, 'south': 35.10, 'west': -90.10, 'tolerance': 130},  # Memphis, TN
    '8':  {'north': 36.21, 'east': -86.73, 'south': 36.11, 'west': -86.83, 'tolerance': 135},  # Nashville, TN
    '9':  {'north': 38.30, 'east': -85.71, 'south': 38.20, 'west': -85.81, 'tolerance': 115},  # Louisville, KY
    '10': {'north': 38.68, 'east': -90.15, 'south': 38.58, 'west': -90.25, 'tolerance': 135},  # St. Louis, MO
    '11': {'north': 39.82, 'east': -86.11, 'south': 39.72, 'west': -86.21, 'tolerance': 140},  # Indianapolis, IN
    '12': {'north': 40.75, 'east': -73.90, 'south': 40.65, 'west': -74.00, 'tolerance': 130},  # New York, NY
    '13': {'north': 42.41, 'east': -71.01, 'south': 42.31, 'west': -71.11, 'tolerance': 100},  # Boston, MA
}


# -- Graph helpers ---------------------------------------------------------------
def download_and_save_graph(area_of_interest, north, east, south, west, filename, tolerance=130):
    roads_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    G = ox.graph_from_bbox(bbox=tuple([west, south, east, north]), retain_all=False, custom_filter=roads_filter, network_type='drive')
    G_proj = ox.project_graph(G.copy())
    G = ox.consolidate_intersections(G_proj, tolerance=tolerance, rebuild_graph=True, dead_ends=True)
    G_di = ox.convert.to_digraph(G, weight='length')
    G = nx.MultiDiGraph(G_di)
    if "simplified" in G.graph:
        del G.graph["simplified"]
        
    # B. Prevent the "TypeError: unhashable type: 'list'" crash during edge stitching
    for u, v, key, data in G.edges(keys=True, data=True):
        for attr_key, attr_value in data.items():
            if isinstance(attr_value, list):
                data[attr_key] = str(attr_value)
                
    # C. Aggressively remove degree-2 nodes, ignoring speed limit/name changes
    G = ox.simplify_graph(G, edge_attrs_differ=[])
    G = ox.project_graph(G, to_crs="EPSG:4326")
    return G


def calculate_graph_area(G):
    """Return convex-hull area of the graph's nodes in km2."""
    G_proj    = ox.project_graph(G)
    gdf_nodes = ox.graph_to_gdfs(G_proj, edges=False)
    all_points  = MultiPoint(gdf_nodes.geometry.tolist())
    convex_hull = all_points.convex_hull
    return convex_hull.area / 1_000_000


def aoi_to_filename(aoi: str) -> str:
    """Convert an area-of-interest string to a safe filename stem."""
    stem = aoi.lower()
    stem = re.sub(r"[,.\s]+", "_", stem)   # spaces / commas -> underscore
    stem = re.sub(r"_+", "_", stem)         # collapse repeated underscores
    stem = stem.strip("_")
    return stem + ".pickle"



def plot_graph_latlon(G, aoi, north, east, south, west, png_path: str):
    """
    Plot using GeoDataFrames on a white background.
    x-axis = Longitude, y-axis = Latitude.
    Bounds are set explicitly from the instance bbox.
    Saved at dpi=300.
    """
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # Plot edges (black) and nodes (red)
    gdf_edges.plot(ax=ax, linewidth=0.8, color='black')
    gdf_nodes.plot(ax=ax, color='red', markersize=20)

    # Axis labels
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)

    # Title with city name and bbox info
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    ax.set_title(
        f"{aoi}  (nodes={n_nodes}, edges={n_edges})\n"
        f"Lat [{south:.3f}, {north:.3f}] | Lon [{west:.3f}, {east:.3f}]",
        fontsize=12, fontweight="bold",
    )

    ax.set_facecolor('white')
    ax.tick_params(colors='black')

    # Fix axis bounds to exactly the instance bbox
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Plot saved -> {os.path.basename(png_path)}")


# -- Main ------------------------------------------------------------------------
def main():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} instances from {EXCEL_PATH}")
    print("=" * 65)

    for _, row in df.iterrows():
        sno = int(row["SNO"])
        aoi = str(row["AOI"]).strip()

        # Look up hardcoded coordinates by SNO key
        coords = INSTANCE_COORDS[str(sno)]
        north     = coords['north']
        east      = coords['east']
        south     = coords['south']
        west      = coords['west']
        tolerance = coords['tolerance']

        filename = aoi_to_filename(aoi)
        pkl_path = os.path.join(OUTPUT_DIR, filename)
        png_path = pkl_path.replace(".pickle", ".png")

        print(f"\n[{sno:02d}/13]  {aoi}")
        print(f"   bbox  N={north}  E={east}  S={south}  W={west}")
        print(f"   tol={tolerance}  ->  {filename}")

        # Skip if already computed
        if os.path.exists(pkl_path):
            print(f"   [OK] Already exists - skipping download.")
            with open(pkl_path, "rb") as f:
                G = pickle.load(f)
        else:
            try:
                G = download_and_save_graph(
                    aoi, north, east, south, west,
                    filename, tolerance=tolerance
                )
                with open(pkl_path, "wb") as f:
                    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"   [OK] Graph saved -> {filename}")
            except Exception as exc:
                print(f"   [FAILED]: {exc}")
                continue

        n_nodes  = G.number_of_nodes()
        n_edges  = G.number_of_edges()
        area_km2 = calculate_graph_area(G)
        print(f"   Nodes={n_nodes}  Edges={n_edges}  Area~{area_km2:.1f} km2")

        # Save plot (always regenerate so it reflects the current graph)
        plot_graph_latlon(
            G,
            aoi=aoi,
            north=north, east=east, south=south, west=west,
            png_path=png_path,
        )

    print("\n" + "=" * 65)
    print(f"Done. All graphs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
