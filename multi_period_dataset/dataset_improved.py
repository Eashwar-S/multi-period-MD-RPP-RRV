import osmnx as ox
import networkx as nx
import pickle
import math
import matplotlib.pyplot as plt
import pandas as pd
import openmeteo_requests
import requests_cache
import os
import numpy as np
from datetime import datetime, timedelta
from retry_requests import retry


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on the Earth (specified in decimal degrees)
    using the Haversine formula.
    
    Parameters:
        lat1, lon1: Latitude and Longitude of point 1 (in degrees)
        lat2, lon2: Latitude and Longitude of point 2 (in degrees)
        
    Returns:
        Distance in meters.
    """
    # Earth radius in meters
    R = 6371000  
    # Convert coordinates from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def spatial_sample_icy_roads(G, road_conditions, min_distance):
    """
    Applies spatial sampling to the icy roads in the road_conditions dictionary,
    ensuring that selected icy roads are at least min_distance apart.
    
    Parameters:
        G (networkx.Graph): The road network graph. Assumes that each node has 'x' and 'y'
                            attributes representing latitude and longitude respectively.
        road_conditions (dict): Dictionary mapping edge (u, v) tuples to condition strings.
        min_distance (float): The minimum distance (in meters) that must separate the selected icy roads.
    
    Returns:
        dict: A new road_conditions dictionary where icy road segments that are too close
              to a previously selected icy road are set to 'normal'.
    """
    # Extract icy edges along with their midpoints (lat, lon)
    icy_edges = []
    for edge, condition in road_conditions.items():
        if condition == 'icy' or condition == 'wet_freezing' or condition == 'snowy' or condition == 'wet':
            u, v = edge
            # Get node coordinates; assuming they are stored under 'x' (latitude) and 'y' (longitude)
            lat_u = G.nodes[u].get('x')
            lon_u = G.nodes[u].get('y')
            lat_v = G.nodes[v].get('x')
            lon_v = G.nodes[v].get('y')
            if None not in (lat_u, lon_u, lat_v, lon_v):
                # Compute the midpoint of the road segment
                mid_lat = (lat_u + lat_v) / 2
                mid_lon = (lon_u + lon_v) / 2
                icy_edges.append((edge, (mid_lat, mid_lon)))
    
    # Greedy selection: select an icy road, then skip any others that are too close.
    selected_edges = []
    selected_midpoints = []
    
    for edge, midpoint in icy_edges:
        if not selected_midpoints:
            selected_edges.append(edge)
            selected_midpoints.append(midpoint)
        else:
            too_close = False
            for sel_mid in selected_midpoints:
                dist = haversine_distance(midpoint[0], midpoint[1], sel_mid[0], sel_mid[1])
                print(f"Distance between {midpoint} and {sel_mid}: {dist}")
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                selected_edges.append(edge)
                selected_midpoints.append(midpoint)
    
    # Update road_conditions: Only keep the icy label if the edge is in the selected set;
    # otherwise, mark it as 'normal'
    sampled_road_conditions = {}
    for edge, condition in road_conditions.items():
        if condition == 'icy' and edge not in selected_edges:
            sampled_road_conditions[edge] = 'normal'
        else:
            sampled_road_conditions[edge] = condition
            
    return sampled_road_conditions

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
    # nodes_dict = {id: (n_data['y'], n_data['x']) for id, n_data in G.nodes(data=True)}
    nodes_dict = {}
    check_list = []
    for id, n_data in G.nodes(data=True):
        nodes_dict[id] = [n_data['y'], n_data['x'], []]
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
    # print(G.edges(data=True))
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

    for k, [lat, lon, edge_list] in nodes_dict.items():
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

        num_days = len(temperature_max)
        edge_list_repeated = [edge_list] * num_days

        daily_data = {
            "node_id": k,
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(days=1),
                inclusive="left"
            ),
            "edge_list": edge_list_repeated,
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
        # for data in daily_data.items():
        #     print(data)
        df = pd.DataFrame(data=daily_data)
        all_data.append(df)

    # Combine all dataframes
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df


def classify_road_conditions(weather_history_df, current_date, node_pairs):
    """
    Classifies road conditions based on weather history and current conditions.
    
    Parameters:
    -----------
    weather_history_df : pd.DataFrame
        DataFrame containing historical weather data with columns:
        ['date', 'node_id', 'temperature_min', 'temperature_max', 'temperature_mean',
         'precipitation_sum', 'snowfall_sum', 'relative_humidity_min',
         'relative_humidity_max']
    current_date : datetime
        The date for which to classify road conditions
    node_pairs : list
        List of (u, v) node pairs representing road segments
        
    Returns:
    --------
    dict
        Dictionary mapping (u, v) pairs to road condition classifications:
        'icy': High risk of ice
        'wet_freezing': Wet roads at risk of freezing
        'snowy': Snow covered
        'wet': Wet but not freezing
        'normal': Normal conditions
    """
    # Constants for classification
    FREEZING_TEMP = 0.0  # °C
    MELTING_TEMP = 1.0   # °C
    HIGH_TEMP = 5.0      # °C
    SNOW_THRESHOLD = 1.0  # mm
    RAIN_THRESHOLD = 0.1  # mm
    HIGH_HUMIDITY = 80.0  # %
    
    # Initialize road conditions dictionary
    road_conditions = {}

    # # Normalize dates to remove time component
    # weather_history_df['date'] = weather_history_df['date'].dt.normalize()
    
    # # Get the last 3 days of weather data
    # date_range = pd.date_range(end=current_date, periods=3, freq='D')
    # print(f"Date Range for Classification: {date_range}")
    weather_history_df['date'] = pd.to_datetime(weather_history_df['date'])
    
    # Step 2: Remove timezone information (if any) and normalize to remove time component
    if weather_history_df['date'].dt.tz is not None:
        weather_history_df['date'] = weather_history_df['date'].dt.tz_convert(None)
    weather_history_df['date'] = weather_history_df['date'].dt.normalize()
    
    # Debug: Check the first few entries after normalization
    print("First few entries in 'weather_history_df' after normalization:")
    print(weather_history_df.head())
    
    # Step 3: Create date_range as normalized dates
    date_range = pd.date_range(end=current_date, periods=3, freq='D').normalize()
    print(f"\nDate Range for Classification: {date_range}")
    historical_data = weather_history_df[weather_history_df['date'].isin(date_range)]
    print(f"Historical Data Rows: {len(historical_data)}")

    if historical_data.empty:
        print("No historical data available for the specified date range.")
    # Group by node_id to get weather history for each node
    node_weather = historical_data.groupby('node_id')
    
    for u, v in node_pairs:
        # Get weather conditions for both nodes of the road segment
        conditions_u = node_weather.get_group(u) if u in node_weather.groups else None
        conditions_v = node_weather.get_group(v) if v in node_weather.groups else None
        
        if conditions_u is None or conditions_v is None:
            road_conditions[(u, v)] = 'normal'
            continue
        
        # Average conditions between the two nodes
        def get_avg_conditions(df_u, df_v, date):
            day_u = df_u[df_u['date'] == date].iloc[0] if not df_u[df_u['date'] == date].empty else None
            day_v = df_v[df_v['date'] == date].iloc[0] if not df_v[df_v['date'] == date].empty else None
            
            if day_u is None or day_v is None:
                return None
            
            return {
                'temp_min': (day_u['temperature_min'] + day_v['temperature_min']) / 2,
                'temp_max': (day_u['temperature_max'] + day_v['temperature_max']) / 2,
                'temp_mean': (day_u['temperature_mean'] + day_v['temperature_mean']) / 2,
                'precip': (day_u['precipitation_sum']/24 + day_v['precipitation_sum']/24) / 2,
                'snow': (day_u['snowfall_sum'] + day_v['snowfall_sum']) / 2,
                'humidity_min': (day_u['relative_humidity_min'] + day_v['relative_humidity_min']) / 2,
                'humidity_max': (day_u['relative_humidity_max'] + day_v['relative_humidity_max']) / 2
            }
        
        # Get conditions for the past three days
        day_conditions = []
        for date in date_range:
            cond = get_avg_conditions(conditions_u, conditions_v, date)
            if cond:
                day_conditions.append(cond)
        
        # print(f"Road Segment ({u}, {v}) - Day Conditions: {day_conditions}")

        if len(day_conditions) < 1:
            road_conditions[(u, v)] = 'normal'
            continue
            
        # Current conditions (most recent day)
        current = day_conditions[-1]
        
        # Classification logic
        if current['snow'] > SNOW_THRESHOLD:
            # Fresh snow on the road
            road_conditions[(u, v)] = 'snowy'
            
        elif len(day_conditions) >= 2:
            prev = day_conditions[-2]
            
            # Check for melting and refreezing conditions
            if (prev['snow'] > SNOW_THRESHOLD and 
                current['temp_max'] > MELTING_TEMP and 
                current['temp_min'] < FREEZING_TEMP):
                road_conditions[(u, v)] = 'icy'
                
            # Check for wet roads that might freeze
            elif (prev['precip'] > RAIN_THRESHOLD and 
                  current['temp_min'] < FREEZING_TEMP):
                road_conditions[(u, v)] = 'wet_freezing'
                
            # Check for melting snow
            elif (prev['snow'] > SNOW_THRESHOLD and 
                  current['temp_mean'] > MELTING_TEMP):
                road_conditions[(u, v)] = 'wet'
                
            # Black ice conditions
            elif (prev['precip'] > 0 and 
                  prev['temp_max'] > FREEZING_TEMP and 
                  current['temp_min'] < FREEZING_TEMP and 
                  current['humidity_max'] > HIGH_HUMIDITY):
                road_conditions[(u, v)] = 'icy'
                
            else:
                road_conditions[(u, v)] = 'normal'
                
        else:
            # Simple current-conditions check if no history
            if current['temp_min'] < FREEZING_TEMP and current['precip'] > 0:
                road_conditions[(u, v)] = 'wet_freezing'
            elif current['temp_mean'] < FREEZING_TEMP and current['humidity_max'] > HIGH_HUMIDITY:
                road_conditions[(u, v)] = 'icy'
            elif current['precip'] > RAIN_THRESHOLD:
                road_conditions[(u, v)] = 'wet'
            else:
                road_conditions[(u, v)] = 'normal'
    
    # print(f"Final Road Conditions: {road_conditions}")

    return road_conditions

def create_daily_icy_road_visualizations(
    graph_file: str,
    weather_data_file: str,
    start_date: str,
    end_date: str,
    output_dir: str = "visualizations",
    figsize: tuple = (15, 15)
):
    """
    Creates daily visualizations of road conditions with advanced classification.
    """
    
    # Color scheme for different road conditions
    CONDITION_COLORS = {
        'icy': 'red',
        'wet_freezing': 'orange',
        'snowy': 'lightblue',
        'wet': 'blue',
        'normal': 'black'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the graph
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        
        # Ensure the graph is a MultiGraph
        if not isinstance(G, nx.MultiGraph):
            G = nx.MultiGraph(G)
        
        # Load and preprocess weather data
        weather_df = pd.read_csv(weather_data_file)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Get all unique node pairs (road segments)
        node_pairs = [(u, v) for u, v, k in G.edges(keys=True)]
        
        # Process each day
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_dt:
            print(f"\nProcessing {current_date.date()}")
            
            # Classify road conditions
            road_conditions = classify_road_conditions(
                weather_df, 
                current_date, 
                node_pairs
            )
        
            min_distance = 5000  # Minimum distance between icy roads
            road_conditions = spatial_sample_icy_roads(G, road_conditions, min_distance)

            # Create visualization
            edge_colors = []
            condition_counts = {condition: 0 for condition in CONDITION_COLORS}
            
            for u, v, k in G.edges(keys=True):
                # print(f'road conditions: {(u, v), road_conditions[(u, v)]}')
                condition = road_conditions.get((u, v), 'normal')
                edge_colors.append(CONDITION_COLORS[condition])
                condition_counts[condition] += 1
            
            # Create the plot
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
            
            # Add title with statistics
            total_edges = G.number_of_edges()
            title = f'Road Conditions - {current_date.strftime("%Y-%m-%d")}\n'
            for condition, count in condition_counts.items():
                if count > 0:
                    percentage = (count / total_edges) * 100
                    title += f'{condition.replace("_", " ").title()}: {count} ({percentage:.1f}%)\n'
            
            plt.title(title, fontsize=16, pad=20)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, alpha=0.6, label=condition.replace('_', ' ').title())
                for condition, color in CONDITION_COLORS.items()
                if condition_counts[condition] > 0
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1, 0.98))
            
            # Display the plot
            print("Close the window to save the image and continue...")
            plt.title(title, fontsize=16, pad=20)
            output_file = os.path.join(output_dir, f'road_conditions_{current_date.strftime("%Y-%m-%d")}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            # plt.show()
            
            # Save the visualization
            
            # plt.close()
            
            print(f"Saved to: {output_file}")
            
            current_date += timedelta(days=1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    



# visualizations = create_daily_icy_road_visualizations(
#     graph_file="college_park.pickle",
#     weather_data_file="college_park_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/college_park"
# )

#kansas
# visualizations = create_daily_icy_road_visualizations(
#     graph_file="kansas.pickle",
#     weather_data_file="kansas_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/kansas"
# )

#friona
# visualizations = create_daily_icy_road_visualizations(
#     graph_file="friona.pickle",
#     weather_data_file="friona_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/friona"
# )

#erie
# visualizations = create_daily_icy_road_visualizations(
#     graph_file="erie.pickle",
#     weather_data_file="erie_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/erie"
# )

#salt lake city
# visualizations = create_daily_icy_road_visualizations(
#     graph_file="salt_lake.pickle",
#     weather_data_file="salt_lake_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/salt_lake"
# )

#washington
# place = "washington"
# visualizations = create_daily_icy_road_visualizations(
#     graph_file= place + ".pickle",
#     weather_data_file= place + "_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/" + place
# )

# new_york
place = "new_york"
graph_name = place + ".pickle"
nodes_dict = load_and_visualize_graph(graph_name)
print(nodes_dict)
start_date = "2025-01-01"
end_date = "2025-01-08"
weather_data = extract_weather_data(nodes_dict, start_date, end_date)
weather_data.to_csv(place + "_jan2025.csv", index=False)

# visualizations = create_daily_icy_road_visualizations(
#     graph_file= place + ".pickle",
#     weather_data_file= place + "_jan2025.csv",
#     start_date="2025-01-01",
#     end_date="2025-01-08",
#     output_dir="road_condition_visualizations/" + place
# )