import os
import osmnx as ox
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
from shapely.geometry import MultiPoint
from scipy.spatial import KDTree
import numpy as np

def extract_weather_data(nodes_dict, start_date, end_date, timezone="America/New_York"):
	"""
	Extracts essential weather data optimized for predicting road icing and black ice 
	from Open-Meteo for given locations.
	"""
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
	retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
	openmeteo = openmeteo_requests.Client(session=retry_session)

	# API parameters - Strictly parameters relevant to road icing/thawing
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"daily": [
			"temperature_2m_max",
			"temperature_2m_min",
			"temperature_2m_mean",
			"dew_point_2m_mean",
			"dew_point_2m_max",
			"dew_point_2m_min",
			"relative_humidity_2m_mean",
			"precipitation_sum",
			"rain_sum",
			"snowfall_sum",
			"wind_speed_10m_max",
			"shortwave_radiation_sum",
			"cloud_cover_mean"
		],
		"timezone": timezone,
		"start_date": start_date,
		"end_date": end_date
	}

	all_data = []

	
	for k, [lat, lon, edge_list, street_count] in nodes_dict.items():

		params["latitude"] = lat
		params["longitude"] = lon

		responses = openmeteo.weather_api(url, params=params)
		response = responses[0]
		daily = response.Daily()

		# Extract daily variables (must follow same order as params["daily"])
		temperature_max = daily.Variables(0).ValuesAsNumpy()
		temperature_min = daily.Variables(1).ValuesAsNumpy()
		temperature_mean = daily.Variables(2).ValuesAsNumpy()

		dewpoint_mean = daily.Variables(3).ValuesAsNumpy()
		dewpoint_max = daily.Variables(4).ValuesAsNumpy()
		dewpoint_min = daily.Variables(5).ValuesAsNumpy()

		humidity_mean = daily.Variables(6).ValuesAsNumpy()

		precipitation_sum = daily.Variables(7).ValuesAsNumpy()
		rain_sum = daily.Variables(8).ValuesAsNumpy()
		snowfall_sum = daily.Variables(9).ValuesAsNumpy()

		wind_speed_10m_max = daily.Variables(10).ValuesAsNumpy()
		shortwave_radiation_sum = daily.Variables(11).ValuesAsNumpy()
		cloud_cover_mean = daily.Variables(12).ValuesAsNumpy()

		num_days = len(temperature_max)

		edge_list_repeated = [edge_list] * num_days
		street_count_repeated = [street_count] * num_days

		daily_data = {
			"node_id": k,
			"date": pd.date_range(
				start=pd.to_datetime(daily.Time(), unit="s", utc=True),
				end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
				freq=pd.Timedelta(days=1),
				inclusive="left"
			),

			"street_count": street_count_repeated,
			"edge_list": edge_list_repeated,

			"latitude": lat,
			"longitude": lon,

			"temperature_max": temperature_max,
			"temperature_min": temperature_min,
			"temperature_mean": temperature_mean,

			"dewpoint_mean": dewpoint_mean,
			"dewpoint_max": dewpoint_max,
			"dewpoint_min": dewpoint_min,

			"humidity_mean": humidity_mean,

			"precipitation_sum": precipitation_sum,
			"rain_sum": rain_sum,
			"snowfall_sum": snowfall_sum,

			"wind_speed_10m_max": wind_speed_10m_max,
			"shortwave_radiation_sum": shortwave_radiation_sum,
			"cloud_cover_mean": cloud_cover_mean
		}

		df = pd.DataFrame(data=daily_data)
		all_data.append(df)

	# Combine all dataframes
	result_df = pd.concat(all_data, ignore_index=True)



	# Make sure rows are ordered by node and date
	result_df = result_df.sort_values(["node_id", "date"]).reset_index(drop=True)

	# Moisture indicator
	result_df["wet_flag"] = (
		(result_df["precipitation_sum"] > 0) |
		(result_df["rain_sum"] > 0) |
		(result_df["snowfall_sum"] > 0) |
		(result_df["humidity_mean"] >= 75)
	).astype(int)

	# Daily severity score
	result_df["severity"] = (
		(result_df["temperature_min"] <= 0).astype(int) +
		(result_df["temperature_min"] <= result_df["dewpoint_mean"]).astype(int) +
		(result_df["humidity_mean"] >= 75).astype(int) +
		result_df["wet_flag"]
	)

	np.random.seed(42)  # For reproducible dataset generation
	daily_noise = np.random.uniform(-0.5, 0.5, len(result_df))
	prob_threshold = np.random.uniform(0.0, 1.0, len(result_df))

	# Deterministic local cooling modifier from topology + unobserved real-world noise
	result_df["delta_local"] = np.select(
		[
			result_df["street_count"] <= 2,
			result_df["street_count"] == 3,
			result_df["street_count"] >= 4
		],
		[
			0.0 + daily_noise * 0.2,
			0.3 + daily_noise * 0.3,
			0.6 + daily_noise * 0.4
		],
		default=0.0
	)

	# Severity-based eligibility (Probabilistic to avoid GNN bias on static node attributes)
	result_df["eligible"] = np.select(
		[
			result_df["severity"] <= 1,
			result_df["severity"] == 2,
			result_df["severity"] >= 3
		],
		[
			0,
			# Severity 2: 80% chance if street_count >= 4, else 40% chance
			np.where(result_df["street_count"] >= 4, (prob_threshold < 0.8).astype(int), (prob_threshold < 0.4).astype(int)),
			# Severity >= 3: 95% chance if street_count >= 3, else 70% chance
			np.where(result_df["street_count"] >= 3, (prob_threshold < 0.95).astype(int), (prob_threshold < 0.7).astype(int))
		],
		default=0
	)

	# Pavement temperature proxy
	result_df["pavement_temp"] = (
		result_df["temperature_min"] - 2.0 - result_df["delta_local"]
	)

	# Base Jang-style formation rule: ice forms today
	result_df["icy_base"] = (
		(result_df["eligible"] == 1) &
		(result_df["pavement_temp"] <= 0) &
		(result_df["pavement_temp"] <= result_df["dewpoint_mean"]) &
		(result_df["wet_flag"] == 1)
	).astype(int)

	# Calculate icy_label sequentially per node using continuous ice thickness
	icy_labels = np.zeros(len(result_df), dtype=int)
	ice_thickness = np.zeros(len(result_df), dtype=float)
	prev_node = None
	prev_thick = 0.0
	
	node_ids = result_df["node_id"].values
	icy_bases = result_df["icy_base"].values
	temp_max = result_df["temperature_max"].values
	
	for i in range(len(result_df)):
		node = node_ids[i]
		if node != prev_node:
			prev_thick = 0.0
			prev_node = node
		
		# Create deterministic variability per node (0.0 to 0.99)
		try:
			variability = (int(node) % 100) / 100.0
		except (ValueError, TypeError):
			variability = 0.5
		
		# Base ice formation condition
		if icy_bases[i] == 1:
			# Ice forms or thickens
			current_thick = max(1.0, prev_thick + 1.0)
		else:
			# If above freezing, ice melts based on temperature + node variability
			if temp_max[i] > 0:
				# Melt rate: e.g. at 5°C max, melts between 0.5 and 2.0 per day
				melt_rate = temp_max[i] * (0.1 + 0.3 * variability)
				current_thick = max(0.0, prev_thick - melt_rate)
			else:
				current_thick = prev_thick
				
		icy_labels[i] = 1 if current_thick > 0.01 else 0
		ice_thickness[i] = current_thick
		prev_thick = current_thick
		
	result_df["icy_label"] = icy_labels
	result_df["ice_thickness"] = ice_thickness
	return result_df


def get_nodes_dict(G):
    nodes_dict = {}
    check_list = []
    for id, n_data in G.nodes(data=True):
        # print(f'node id - {id}, node data - {n_data}')
        nodes_dict[id] = [n_data['y'], n_data['x'], [], n_data['street_count']]
        for n1, n2, _ in G.edges(data=True):
            if id == n1:
                nodes_dict[id][2].append((n2))
            if id == n2:
                nodes_dict[id][2].append((n1))
            if id == 0:
                if id == n1:
                    check_list.append(n2)
                if id == n2:
                    check_list.append(n1)
        nodes_dict[id][2] = list(set(nodes_dict[id][2]))

    return nodes_dict


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
	G_proj = ox.project_graph(G)
	gdf_nodes = ox.graph_to_gdfs(G_proj, edges=False)
	all_points = MultiPoint(gdf_nodes.geometry.tolist())
	convex_hull = all_points.convex_hull
	area_sq_meters = convex_hull.area
	area_sq_km = area_sq_meters / 1_000_000
	return area_sq_km



# --- Dataset 1 ---
# aoI = "College Park, Maryland, US"
north, east, south, west = 39.00164, -76.90541, 38.91214, -77.02000
aoI = "Arlington, Texaas, US"
north, east, south, west = 32.77486, -97.04704, 32.67753, -97.17201
graph_name = "college_park.pickle"

G = download_and_save_graph(aoI, north, east, south, west, graph_name, tolerance=120)

print(f"Number of edges: {G.number_of_edges()}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f'Graph area is {calculate_graph_area(G)} km^2')

fig, ax = ox.plot_graph(G, node_size=30, node_color="r", edge_linewidth=0.5)
# plt.show()
start_date = "2026-01-21"
end_date = "2026-02-06"
nodes_dict = get_nodes_dict(G)
weather_data = extract_weather_data(nodes_dict, start_date, end_date)

os.makedirs("graph_data", exist_ok=True)
weather_data.to_csv("graph_data/" + aoI + "_Jan_2026.csv", index=False)


file_path = "graph_data/" + aoI + "_Jan_2026.csv"
# file_path = "graph_data/Arlington, Texaas, US_Jan_2026.csv"
df = pd.read_csv(file_path)


daily_icy_counts = df.groupby("date")["icy_label"].sum()
print(daily_icy_counts)

plt.figure(figsize=(10,6))

plt.bar(
    daily_icy_counts.index.astype(str),
    daily_icy_counts.values
)

plt.xlabel("Date")
plt.ylabel("Number of Icy Edges")
plt.title("Number of Icy Edges per Day")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()