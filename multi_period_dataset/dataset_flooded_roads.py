import osmnx as ox
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
from matplotlib.patches import Patch
import os


def download_and_save_graph(area_of_interest, north, east, south, west, filename):
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
    
    G = ox.consolidate_intersections(ox.project_graph(G.copy()), tolerance=100)
    # print(G.nodes(data=True))
    # print("Current CRS:", G.graph["crs"])
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
    nodes_dict = {id: (n_data['y'], n_data['x']) for id, n_data in G.nodes(data=True)}

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
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "snowfall_sum", "relative_humidity_2m_max", "relative_humidity_2m_min", "dewpoint_2m_max", "dewpoint_2m_min"],
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date
    }

    all_data = []

    for k, (lat, lon) in nodes_dict.items():
        params["latitude"] = lat
        params["longitude"] = lon

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        daily = response.Daily()
        temperature_max = daily.Variables(0).ValuesAsNumpy()
        temperature_min = daily.Variables(1).ValuesAsNumpy()
        temperature_mean = daily.Variables(2).ValuesAsNumpy()
        precipitation_sum = daily.Variables(3).ValuesAsNumpy()
        snowfall_sum = daily.Variables(4).ValuesAsNumpy()
        relative_humidity_max = daily.Variables(5).ValuesAsNumpy()
        relative_humidity_min = daily.Variables(6).ValuesAsNumpy()
        dewpoint_max = daily.Variables(7).ValuesAsNumpy()
        dewpoint_min = daily.Variables(8).ValuesAsNumpy()

        daily_data = {
            "node_id": k,
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(days=1),
                inclusive="left"
            ),
            "latitude": lat,
            "longitude": lon,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "temperature_mean": temperature_mean,
            "precipitation_sum": precipitation_sum,
            "snowfall_sum": snowfall_sum,
            "relative_humidity_max": relative_humidity_max,
            "relative_humidity_min": relative_humidity_min,
            "dewpoint_max": dewpoint_max,
            "dewpoint_min": dewpoint_min
        }
        df = pd.DataFrame(data=daily_data)
        all_data.append(df)

    # Combine all dataframes
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df

def classify_flooded_road_conditions(weather_df, current_date, node_pairs):
    """
    Classifies road segments as 'flooded' or 'normal' based on multiple weather conditions
    that may lead to flooding. The classification considers the following factors:
    
    1. Heavy Rainfall:
       - If precipitation_sum >= 10.0 mm, a large volume of rain may cause flooding.
    
    2. Rapid Snowmelt:
       - If snowfall_sum >= 5.0 mm AND temperature_mean >= 1°C, melting snow may produce
         enough water to flood roads.
    
    3. Saturated Air (High Dewpoint):
       - If (temperature_max - dewpoint_max) <= 2°C, the air is nearly saturated. In such
         conditions, even moderate water input can lead to flooding.
    
    4. Combined Moderate Factors:
       - If precipitation_sum >= 5.0 mm AND snowfall_sum >= 2.0 mm, then the combined water
         from both rain and snow may cause flooding.
    
    Parameters:
        weather_df (pd.DataFrame): DataFrame containing weather data with columns for date,
                                   node_id, precipitation_sum, snowfall_sum, temperature_mean,
                                   temperature_max, and dewpoint_max.
        current_date (datetime): The day for which to perform classification.
        node_pairs (list of tuple): List of (u, v) pairs representing road segments.
        
    Returns:
        dict: Mapping of each road segment (u, v) to either 'flooded' or 'normal'.
    """
    # Define thresholds for each condition
    heavy_rainfall_threshold = 10.0         # mm of precipitation
    heavy_snowfall_threshold = 5.0            # mm of snowfall
    melting_temperature_threshold = 1.0       # °C above which snow may melt rapidly
    dewpoint_diff_threshold = 2.0             # °C difference indicating saturated air
    moderate_precip_threshold = 5.0           # mm of precipitation (moderate)
    moderate_snow_threshold = 2.0             # mm of snowfall (moderate)
    
    def check_conditions(row):
        """
        Checks which flooding conditions are met for a given weather record.
        Returns a list of condition tags (empty if none are met).
        """
        conditions = []
        # 1. Heavy Rainfall
        if row.get('precipitation_sum', 0.0) >= heavy_rainfall_threshold:
            conditions.append('heavy_rainfall')
        # 2. Rapid Snowmelt
        if row.get('snowfall_sum', 0.0) >= heavy_snowfall_threshold and row.get('temperature_mean', 0.0) >= melting_temperature_threshold:
            conditions.append('rapid_snowmelt')
        # 3. Saturated Air (High Dewpoint)
        if (row.get('temperature_max', 0.0) - row.get('dewpoint_max', 0.0)) <= dewpoint_diff_threshold:
            conditions.append('saturated_air')
        # 4. Combined Moderate Factors
        if row.get('precipitation_sum', 0.0) >= moderate_precip_threshold and row.get('snowfall_sum', 0.0) >= moderate_snow_threshold:
            conditions.append('combined_moderate')
        return conditions

    conditions_by_segment = {}
    # Filter the weather data for the current day
    day_weather = weather_df[weather_df['date'] == pd.to_datetime(current_date)]
    
    for u, v in node_pairs:
        conditions_triggered = []
        
        # Get weather data for endpoint u
        u_weather = day_weather[day_weather['node_id'] == u]
        if not u_weather.empty:
            conditions_triggered.extend(check_conditions(u_weather.iloc[0]))
        
        # Get weather data for endpoint v
        v_weather = day_weather[day_weather['node_id'] == v]
        if not v_weather.empty:
            conditions_triggered.extend(check_conditions(v_weather.iloc[0]))
        
        # If any condition is triggered at either endpoint, mark the road as flooded.
        if conditions_triggered:
            conditions_by_segment[(u, v)] = 'flooded'
        else:
            conditions_by_segment[(u, v)] = 'normal'
            
    return conditions_by_segment


def create_daily_flooded_road_visualizations(
    graph_file: str,
    weather_data_file: str,
    start_date: str,
    end_date: str,
    output_dir: str = "visualizations",
    figsize: tuple = (15, 15)
):
    """
    Creates daily visualizations of road conditions by classifying roads as flooded or normal.
    Uses a complex classifier that reasons over multiple weather conditions:
      - Heavy Rainfall
      - Rapid Snowmelt
      - Saturated Air (High Dewpoint)
      - Combined Moderate Factors
    
    Parameters:
        graph_file (str): Path to the pickle file containing the road network graph.
        weather_data_file (str): Path to the CSV file with weather data.
        start_date (str): Start date (YYYY-MM-DD) for visualization.
        end_date (str): End date (YYYY-MM-DD) for visualization.
        output_dir (str): Directory where visualization images will be saved.
        figsize (tuple): Size of the generated figures.
    """
    # Define color scheme for visualization
    CONDITION_COLORS = {
        'flooded': 'blue',
        'normal': 'black'
    }
    
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the road network graph.
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        if not isinstance(G, nx.MultiGraph):
            G = nx.MultiGraph(G)
        
        # Load and preprocess weather data.
        weather_df = pd.read_csv(weather_data_file)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Create a list of unique road segments (node pairs).
        node_pairs = [(u, v) for u, v, k in G.edges(keys=True)]
        
        # Process each day from start_date to end_date.
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_dt:
            print(f"\nProcessing {current_date.date()}")
            
            # Classify road conditions using the complex flooding criteria.
            road_conditions = classify_flooded_road_conditions(
                weather_df, 
                current_date, 
                node_pairs
            )
            
            # Prepare edge colors and count condition occurrences.
            edge_colors = []
            condition_counts = {condition: 0 for condition in CONDITION_COLORS}
            
            for u, v, k in G.edges(keys=True):
                condition = road_conditions.get((u, v), 'normal')
                edge_colors.append(CONDITION_COLORS[condition])
                condition_counts[condition] += 1
            
            # Plot the graph using OSMnx.
            fig, ax = ox.plot_graph(
                G,
                node_size=20,
                node_color='gray',
                node_alpha=0.4,
                edge_color=edge_colors,
                edge_linewidth=1.5,
                edge_alpha=0.6,
                bgcolor='white',
                show=False,
                close=False,
                figsize=figsize
            )
            
            # Build a title with daily statistics.
            total_edges = G.number_of_edges()
            title = f'Flooded Road Conditions - {current_date.strftime("%Y-%m-%d")}\n'
            for condition, count in condition_counts.items():
                if count > 0:
                    percentage = (count / total_edges) * 100
                    title += f'{condition.title()}: {count} ({percentage:.1f}%)\n'
            
            plt.title(title, fontsize=16, pad=20)
            
            # Add legend to the plot.
            legend_elements = [
                Patch(facecolor=color, alpha=0.6, label=condition.title())
                for condition, color in CONDITION_COLORS.items()
                if condition_counts[condition] > 0
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.98))
            
            # Save the visualization.
            output_file = os.path.join(output_dir, f'flooded_road_conditions_{current_date.strftime("%Y-%m-%d")}.png')
            print("Close the window to save the image and continue...")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            # Uncomment plt.show() to display interactively.
            # plt.show()
            plt.close()
            
            print(f"Saved to: {output_file}")
            current_date += timedelta(days=1)
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")





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

# threshold = 200
# aoI = "Salt lake city, Utah, US"
# north, east, south, west = 40.7846, -111.7860, 40.6257, -112.0928
# graph_name = "salt_lake.pickle"
# download_and_save_graph(aoI, north, east, south, west, graph_name)
# nodes_dict = load_and_visualize_graph(graph_name)
# print(len(nodes_dict))

# start_date = "2025-01-01"
# end_date = "2025-01-08"
# weather_data = extract_weather_data(nodes_dict, start_date, end_date)
# weather_data.to_csv("salt_lake_jan2025.csv", index=False)

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


places = ["kansas", "friona", "salt_lake", "washington", "new_york", "college_park"]
for place in places:
    visualizations = create_daily_flooded_road_visualizations(
        graph_file= place + ".pickle",
        weather_data_file= place + "_jan2025.csv",
        start_date="2025-01-01",
        end_date="2025-01-08",
        output_dir="flooded_road_condition_visualizations/" + place
    )