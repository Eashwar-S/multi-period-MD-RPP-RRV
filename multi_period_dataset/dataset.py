import osmnx as ox
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta


FREEZING_TEMP = 0  # °C
PRECIPITATION_THRESHOLD = 0.1  # mm
SNOWFALL_THRESHOLD = 0  # mm

def download_and_save_graph(area_of_interest, north, east, south, west, threshold, filename):
    """
    Downloads a map as a graph object for a specified area and saves it to a file.
    
    Parameters:
    area_of_interest (str): The area of interest to download the map for.
    north (float): Northern boundary of the bounding box.
    east (float): Eastern boundary of the bounding box.
    south (float): Southern boundary of the bounding box.
    west (float): Western boundary of the bounding box.
    filename (str): The name of the file to save the graph to.
    """
    # Configure OSMnx
    # ox.config(use_cache=True, log_console=True)

    # Define the area in Atlanta (downtown area)
    state = ox.geocode_to_gdf(area_of_interest)
    # roads_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
    roads_filter = '["highway"~"motorway|truck|primary|secondary"]'
    # Downloading the map as a graph object
    G = ox.graph_from_bbox(bbox=tuple([west, south, east, north]), retain_all=False, custom_filter=roads_filter)
    print(G.nodes(data=True))
    
    G = ox.elevation.add_node_elevations_google(G, api_key="AIzaSyDbjBao76hNz8pY3GPnIKUoJ4wBn8FR4j0")

    G = ox.consolidate_intersections(ox.project_graph(G.copy()), tolerance=threshold, rebuild_graph=True, dead_ends=False)


    G = ox.project_graph(G, to_crs="EPSG:4326")
    # print(G.nodes(data=True))
    # Save the graph
    with open(filename, 'wb') as f:
        pickle.dump(G, f)

def load_and_visualize_graph(filename):
    """
    Loads a saved graph from a file and visualizes it as an undirected graph.
    
    Parameters:
    filename (str): The name of the file to load the graph from.
    """
    # Load the graph
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    
    # Extract latitudes and longitudes
    # nodes_dict = {id: (n_data['y'], n_data['x']) for id, n_data in G.nodes(data=True)}
    nodes_dict = {}
    check_list = []
    for id, n_data in G.nodes(data=True):
        nodes_dict[id] = [n_data['y'], n_data['x'], [], n_data['elevation']]
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

    G = ox.project_graph(G)

    # Plot the graph
    fig, ax = ox.plot_graph(G, node_size=30, node_color="r", edge_linewidth=0.5)
    plt.show()

    return nodes_dict


def extract_weather_data(nodes_dict, start_date, end_date, timezone="America/New_York"):
    """
    Extracts temperature, precipitation, snowfall, relative humidity, and dewpoint data from Open-Meteo for given locations.
    
    Parameters:
    nodes_dict (dict): Dictionary of node IDs with their corresponding latitudes and longitudes.
    start_date (str): The start date for the weather data in YYYY-MM-DD format.
    end_date (str): The end date for the weather data in YYYY-MM-DD format.
    timezone (str): The timezone for the weather data.
    
    Returns:
    pd.DataFrame: DataFrame containing the weather data for all locations.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "daily": [
            "temperature_2m_max",       # index 0
            "temperature_2m_min",       # index 1
            "temperature_2m_mean",      # index 2
            "sunshine_duration",        # index 3
            "uv_index_max",             # index 4
            "precipitation_sum",        # index 5
            "snowfall_sum",             # index 6
            "rain_sum",                 # index 7
            "precipitation_hours",      # index 8
            "precipitation_probability_max",  # index 9
            "wind_speed_10m_max",       # index 10
            "wind_gusts_10m_max"        # index 11
        ],
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date
    }

    all_data = []

    for k, [lat, lon, edge_list, elevation] in nodes_dict.items():
        params["latitude"] = lat
        params["longitude"] = lon

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        daily = response.Daily()
        # Extract daily variables using updated indices
        temperature_max = daily.Variables(0).ValuesAsNumpy()
        temperature_min = daily.Variables(1).ValuesAsNumpy()
        temperature_mean = daily.Variables(2).ValuesAsNumpy()
        sunshine_duration = daily.Variables(3).ValuesAsNumpy()
        uv_index_max = daily.Variables(4).ValuesAsNumpy()
        precipitation_sum = daily.Variables(5).ValuesAsNumpy()
        snowfall_sum = daily.Variables(6).ValuesAsNumpy()
        rain_sum = daily.Variables(7).ValuesAsNumpy()
        precipitation_hours = daily.Variables(8).ValuesAsNumpy()
        precipitation_probability_max = daily.Variables(9).ValuesAsNumpy()
        wind_speed_10m_max = daily.Variables(10).ValuesAsNumpy()
        wind_gusts_10m_max = daily.Variables(11).ValuesAsNumpy()

        num_days = len(temperature_max)
        edge_list_repeated = [edge_list] * num_days
        elevation_repeated = [elevation] * num_days

        daily_data = {
            "node_id": k,
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(days=1),
                inclusive="left"
            ),
            "edge_list": edge_list_repeated,
            "elevation": elevation_repeated,
            "latitude": lat,
            "longitude": lon,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "temperature_mean": temperature_mean,
            "sunshine_duration": sunshine_duration,
            "uv_index_max": uv_index_max,
            "precipitation_sum": precipitation_sum,
            "snowfall_sum": snowfall_sum,
            "rain_sum": rain_sum,
            "precipitation_hours": precipitation_hours,
            "precipitation_probability_max": precipitation_probability_max,
            "wind_speed_10m_max": wind_speed_10m_max,
            "wind_gusts_10m_max": wind_gusts_10m_max
        }
        df = pd.DataFrame(data=daily_data)
        all_data.append(df)

    # Combine all dataframes
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

def create_weather_graphs(csv_file, graph_file, start_date, end_date):
    """
    Creates OSMnx graphs for each day between start_date and end_date, highlighting edges based on weather data.
    
    Parameters:
    csv_file (str): Path to the CSV file containing weather data.
    graph_file (str): Path to the pickle file containing the graph.
    start_date (str): The start date for the graphs in YYYY-MM-DD format.
    end_date (str): The end date for the graphs in YYYY-MM-DD format.
    """
    # Load the graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Load the weather data
    weather_data = pd.read_csv(csv_file)
    
    # Convert date column to datetime
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    
    # Generate graphs for each day
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_date:
        # Filter weather data for the current date
        daily_data = weather_data[weather_data['date'] == current_date]
        
        # Create a copy of the graph to modify
        G_copy = G.copy()
        
        nx.set_edge_attributes(G_copy, 'white', 'color')

        # Create a dictionary to store weather data for each node
        node_weather = {}
        for _, row in daily_data.iterrows():
            node_id = row['node_id']
            node_weather[node_id] = row

        # print(node_weather, daily_data)
        # Highlight edges based on average weather data of connecting nodes
        for u, v, data in G_copy.edges(data=True):
            if u in node_weather and v in node_weather:
                u_weather = node_weather[u]
                v_weather = node_weather[v]
                
                # Calculate average weather conditions for the edge
                avg_temp = (u_weather['temperature_mean'] + v_weather['temperature_mean']) / 2
                avg_precip = (u_weather['precipitation_sum'] + v_weather['precipitation_sum']) / 2
                avg_snowfall = (u_weather['snowfall_sum'] + v_weather['snowfall_sum']) / 2
                
                print(avg_temp, avg_precip, avg_snowfall)
                # Determine edge color based on average weather conditions
                if (avg_temp <= FREEZING_TEMP or 
                    (avg_precip >= PRECIPITATION_THRESHOLD or avg_snowfall > SNOWFALL_THRESHOLD)):
                    # print('here')
                    data['color'] = 'red'
                else:
                    data['color'] = 'white'
   
        # print(G.edges(data=True))
        # Plot the graph
        edge_colors = [data['color'] for u, v, data in G_copy.edges(data=True)]
        fig, ax = ox.plot_graph(G_copy, node_size=30, node_color="r", edge_linewidth=0.5, edge_color=edge_colors)
        # plt.title(f"Weather Graph for {current_date.strftime('%Y-%m-%d')}")
        plt.show()
        
        # Save the graph plot
        fig.savefig(f"weather_graph_{current_date.strftime('%Y-%m-%d')}.png")
        
        # Move to the next day
        current_date += timedelta(days=1)

#v1
def create_daily_icy_road_graphs(
    graph_pickle_path: str,
    csv_data_path: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-08",
    # Simple threshold logic: road is "icy" if T_min < 0 and (precip + snowfall) > 0
    temp_min_threshold: float = 0.0,
    precip_snow_threshold: float = 0.0
):
    """
    Loads a precomputed OSMnx graph from `graph_pickle_path`, reads weather data
    from `csv_data_path`, and for each day in [start_date, end_date], highlights 
    edges in red where at least one of the edge's endpoint-nodes meets the 'icy'
    condition. Otherwise, edges are black.

    :param graph_pickle_path: Path to the pickle file containing the OSMnx graph.
    :param csv_data_path: Path to the CSV file with weather data. Must contain
                          at least: ['node_id','date','temperature_min','precipitation_sum','snowfall_sum'].
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    :param temp_min_threshold: Temperature below which roads may freeze (default=0°C).
    :param precip_snow_threshold: If (precipitation_sum + snowfall_sum) exceeds this,
                                  we consider roads possibly icy (default=0 → any snow/precip).
    :return: A dictionary mapping {date_string: (fig, ax)} for each day, where 
             edges in the plotted graph are colored red for icy roads, black otherwise.
    """

    # 1. Load the precomputed graph
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)

    # Make sure your graph has node attributes keyed by OSM id or something consistent
    # with your CSV's "node_id". If not, you may need a mapping from node_id → graph node.

    # 2. Read the CSV weather data
    df = pd.read_csv(csv_data_path, parse_dates=["date"])

    # Filter by date range
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (df["date"] >= start_dt) & (df["date"] < end_dt)
    df = df.loc[mask].copy()

    # For convenience, also add a day-only column (no hour)
    df["day"] = df["date"].dt.floor("D")

    # 3. Group by day so we can easily check icy conditions for each day
    grouped = df.groupby("day")

    # 4. Prepare a dictionary to hold the resulting day-specific graphs or plots
    daily_graph_plots = {}

    # 5. Iterate through each day in the range [start_date, end_date)
    all_days = pd.date_range(start_dt, end_dt, freq="D")
    for day in all_days:
        day_str = day.strftime("%Y-%m-%d")
        # Subset for the day
        if day not in grouped.groups:
            # No data for this day, skip or create a blank plot
            continue

        day_df = grouped.get_group(day)

        # We will mark each node as "icy" or not
        # Create a set of "icy" node_ids
        icy_node_ids = set()
        for _, row in day_df.iterrows():
            node_id = row["node_id"]
            temp_min = row["temperature_min"]
            precip_snow = row["precipitation_sum"] + row["snowfall_sum"]

            # Check if this node meets the icy threshold
            if (temp_min < temp_min_threshold) and (precip_snow > precip_snow_threshold):
                icy_node_ids.add(node_id)

        # Make a copy of the original graph so we can store day-specific color attributes
        # (You can also store them on the original graph and reset them each time if you prefer.)
        G_day = nx.Graph()
        G_day = G.copy(as_view=False)

        # 6. Color edges:
        # If ANY endpoint is in the icy_node_ids, color that edge red. Otherwise black.
        edge_colors = []
        for (u, v, data) in G_day.edges(data=True):
            # If the node_id in the CSV matches u or v, we need a way to match them.
            # This depends on how your graph nodes are named. 
            # Suppose your G node keys are int node_ids that match CSV. 
            # If that's the case, then:
            if (u in icy_node_ids) or (v in icy_node_ids):
                edge_colors.append("red")
            else:
                edge_colors.append("black")

        # 7. Plot this day's graph
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = {node: (G_day.nodes[node]["x"], G_day.nodes[node]["y"]) 
               for node in G_day.nodes if "x" in G_day.nodes[node] and "y" in G_day.nodes[node]}

        # Draw the edges with the color list
        nx.draw_networkx_edges(
            G_day,
            pos=pos,
            edge_color=edge_colors,
            ax=ax,
            width=1
        )
        # Optionally, you can also hide nodes or draw them differently
        nx.draw_networkx_nodes(
            G_day,
            pos=pos,
            ax=ax,
            node_size=5,
            node_color="blue",
            alpha=0.2
        )

        ax.set_title(f"Icy Roads on {day_str}")
        ax.axis("off")
        plt.tight_layout()

        # 8. Store or show the figure
        daily_graph_plots[day_str] = (fig, ax)
        # If you want to save each figure, uncomment:
        # plt.savefig(f"college_park_{day_str}.png", dpi=150)
        # plt.close(fig)

    return daily_graph_plots

# Example usage
# aoI = "Atlanta, Georgia, US"
# north, east, south, west = 33.75482, -84.38607, 33.74547, -84.39697


def create_daily_icy_road_multigraphs(
    graph_pickle_path: str,
    csv_data_path: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-01-08",
    temp_min_threshold: float = 0.0,
    precip_snow_threshold: float = 0.0
):
    """
    Loads a precomputed (possibly directed) OSMnx graph from `graph_pickle_path`,
    converts it to an undirected MultiGraph, reads weather data from `csv_data_path`,
    and for each day in [start_date, end_date], highlights edges in red if at least
    one endpoint node meets 'icy' conditions. Otherwise, edges are black.

    Returns a dict of {date_str: (fig, ax)} for each day.
    """

    # 1. Load the original graph (likely directed, simple, etc.)
    with open(graph_pickle_path, "rb") as f:
        G_orig = pickle.load(f)

    # 2. Convert to an undirected MultiGraph
    #    - If G_orig is directed, to_undirected() merges edges, but
    #      we can keep multiplicities by going to a MultiGraph:
    G_undirected = nx.MultiGraph(nx.to_undirected(G_orig))
    
    # 3. Read the CSV weather data
    df = pd.read_csv(csv_data_path, parse_dates=["date"])

    # If your CSV datetimes are tz-aware and your start/end are tz-naive (or vice versa),
    # ensure they match. For example, if CSV is tz-aware in UTC, do:
    # df["date"] = df["date"].dt.tz_localize(None)

    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt   = pd.to_datetime(end_date, utc=True)

    mask = (df["date"] >= start_dt) & (df["date"] < end_dt)
    df = df.loc[mask].copy()

    # 4. Create a day-only column so we can group by day
    df["day"] = df["date"].dt.floor("D")

    grouped = df.groupby("day")

    # 5. Prepare a dictionary to hold day-specific results
    daily_graph_plots = {}

    # 6. Iterate through each day
    all_days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    for day in all_days:
        day_str = day.strftime("%Y-%m-%d")
        if day not in grouped.groups:
            # No data this day → skip or create empty plot
            continue

        day_df = grouped.get_group(day)

        # Build a set of node_ids that meet the "icy" condition
        icy_node_ids = set()
        for _, row in day_df.iterrows():
            node_id = row["node_id"]
            temp_min = row["temperature_min"]
            precip_snow = row["precipitation_sum"] + row["snowfall_sum"]
            if (temp_min < temp_min_threshold) and (precip_snow > precip_snow_threshold):
                icy_node_ids.add(node_id)

        # 7. Make a copy of the undirected MultiGraph so each day gets its own color attributes
        G_day = nx.MultiGraph()
        G_day.add_nodes_from(G_undirected.nodes(data=True))
        G_day.add_edges_from(G_undirected.edges(data=True, keys=True))
        
        # 8. For coloring edges, if any endpoint is in icy_node_ids → red, else black
        edge_colors = []
        for (u, v, key, data) in G_day.edges(keys=True, data=True):
            if (u in icy_node_ids) or (v in icy_node_ids):
                edge_colors.append("red")
            else:
                edge_colors.append("black")

        # 9. Plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # If each node has 'x'/'y' attributes (as in OSMnx), gather them
        pos = {}
        for node in G_day.nodes:
            node_data = G_day.nodes[node]
            if "x" in node_data and "y" in node_data:
                pos[node] = (node_data["x"], node_data["y"])

        # Draw edges
        nx.draw_networkx_edges(
            G_day,
            pos=pos,
            edge_color=edge_colors,
            ax=ax,
            width=1
        )
        # Draw nodes (optionally)
        nx.draw_networkx_nodes(
            G_day,
            pos=pos,
            ax=ax,
            node_size=5,
            node_color="blue",
            alpha=0.2
        )

        ax.set_title(f"Icy Roads on {day_str} (Undirected MultiGraph)")
        ax.axis("off")
        plt.tight_layout()

        daily_graph_plots[day_str] = (fig, ax)

        # Optionally save:
        # outfile = f"icy_roads_{day_str}.png"
        # fig.savefig(outfile, dpi=150)
        # plt.close(fig)

    return daily_graph_plots



# aoI = "College Park, Maryland, US"
# north, east, south, west = 38.99524, -76.92288, 38.98260, -76.94831
# graph_name = "college_park.pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)


#kansas
# aoI = "Kansas City, Missouri, US"
# north, east, south, west = 39.09623, -94.55658, 39.07304, -94.58988
# graph_name = "kansas.pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv("kansas_jan2025.csv", index=False)



# aoI = "Friona, Texas, US"
# north, east, south, west = 34.64613, -102.71350, 34.63307, -102.73599
# graph_name = "friona.pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv("friona_jan2025.csv", index=False)

# aoI = "Erie, Pennsylvania, US"
# north, east, south, west = 42.15577, -80.01308, 42.07924, -80.13273
# graph_name = "erie.pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv("erie_jan2025.csv", index=False)

threshold = 200
aoI = "Salt lake city, Utah, US"
north, east, south, west = 40.7846, -111.7860, 40.6257, -112.0928
graph_name = "graph_files/salt_lake.pickle"
download_and_save_graph(aoI, north, east, south, west, threshold, graph_name)
nodes_dict = load_and_visualize_graph(graph_name)
print(len(nodes_dict))

start_date = "2025-01-01"
end_date = "2025-01-16"
weather_data = extract_weather_data(nodes_dict, start_date, end_date)
weather_data.to_csv("graph_data/salt_lake_jan2025.csv", index=False)

# threshold = 200
# aoI = "Spokane, Washington, US"
# place = "washington"
# north, east, south, west = 47.7587, -117.1328, 47.5900, -117.5029
# graph_name = place + ".pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv(place + "_jan2025.csv", index=False)


# threshold = 250
# aoI = "Buffalo, New York, US"
# place = "new_york"
# north, east, south, west = 43.0669, -78.4644, 42.7873, -78.9800
# graph_name = place + ".pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv(place + "_jan2025.csv", index=False)


# threshold = 250
# aoI = "Flagstaff, Arizona, US"
# place = "arizona"
# north, east, south, west = 35.22837, -111.58213, 35.17170, -111.68478
# graph_name = place + ".pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv(place + "_jan2025.csv", index=False)


# After saving 

# csv_file = "college_park_jan2025.csv"
# graph_file = "college_park.pickle"
# # create_weather_graphs(csv_file, graph_file, start_date, end_date)
# # print("\nFinal DataFrame:")
# # print(weather_data)

# results = create_daily_icy_road_graphs(
#         graph_pickle_path=graph_file,
#         csv_data_path=csv_file,
#         start_date="2025-01-01",
#         end_date="2025-01-08",
#         temp_min_threshold=0.0,            # if T_min < 0, we suspect icing
#         precip_snow_threshold=0.0          # if there's any precipitation or snowfall, roads are icy
#     )
# for day_str, (fig, ax) in results.items():
#     print(f"Displaying and saving figure for: {day_str}")
    
#     # Display the figure (in some interactive environments it may pop up a window)
#     fig.show()

#     # Save to a PNG file (or any other format supported by matplotlib)
#     filename = f"{day_str}_icy_roads.png"
#     fig.savefig(filename, dpi=300)
#     print(f"Saved: {filename}")

#     # If you don't plan to interact with the figure afterward, you can close it
#     plt.close(fig)