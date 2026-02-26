import networkx as nx
import osmnx as ox
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def convert_edge_weights_to_time(G, speed):
    """
    For each edge in the graph G, add a 'travel_time' attribute equal to edge length / speed.
    Assumes edge attribute 'length' exists.
    """
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / speed
    return G

def compute_coverage(G, radius):
    """
    For each node in G, compute the set of nodes reachable within the given radius (in minutes).
    Returns a dictionary mapping node -> set(covered nodes).
    """
    coverage = {}
    for node in G.nodes():
        # Using Dijkstra's algorithm with the 'travel_time' attribute.
        lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=radius, weight='travel_time')
        coverage[node] = set(lengths.keys())
    return coverage

def compute_coverage_specific_nodes(G, nodes, radius):
    
    coverage = {}
    for node in nodes:
        # Using Dijkstra's algorithm with the 'travel_time' attribute.
        lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=radius, weight='travel_time')
        coverage[node] = set(lengths.keys())
    return coverage

# def select_depots_optimized(G, battery_capacity, speed, alpha=1.0, beta=1.0):
#     """
#     Greedy heuristic to select depot nodes using a modified greedy score that combines
#     service coverage (round-trip constraint) and inter-depot connectivity (one-way reachability).
    
#     Parameters:
#     - G: A networkx graph (with edge attribute 'length').
#     - battery_capacity: Total allowed travel time (minutes) per trip.
#     - speed: Vehicle speed (consistent with edge length units) to compute travel time.
#     - alpha: Weight for service coverage gain.
#     - beta: Weight for connectivity benefit.
    
#     Returns:
#     - selected: set of depot nodes.
#     """
#     # Convert edge lengths to travel time
#     G = convert_edge_weights_to_time(G, speed)
    
#     # Define radii for service and connectivity
#     radius_service = battery_capacity / 2.0  # round-trip constraint for service
#     radius_connect = battery_capacity         # one-way connectivity between depots
    
#     # Compute coverage sets for each node
#     coverage_service = compute_coverage(G, radius_service)
#     coverage_connect = compute_coverage(G, radius_connect)
    
#     all_nodes = set(G.nodes())
#     selected = set()
#     covered = set()
    
#     # Greedy selection using modified score:
#     # score(u) = alpha * (# new nodes in service coverage) + beta * (# already selected nodes in connectivity coverage)
#     while covered != all_nodes:
#         best_node = None
#         best_score = float('-inf')
#         for node in G.nodes():
#             if node in selected:
#                 continue
#             # Service gain: number of nodes newly covered by node's service coverage
#             service_gain = len(coverage_service[node] - covered)
#             # Connectivity benefit: number of already selected depots within connectivity coverage
#             connectivity_benefit = len(coverage_connect[node] & selected)
#             score = alpha * service_gain + beta * connectivity_benefit
#             if score > best_score:
#                 best_score = score
#                 best_node = node
#         if best_node is None:
#             break  # safety break if no progress can be made
#         selected.add(best_node)
#         covered |= coverage_service[best_node]
    
#     # Optional post-processing: Ensure each depot has at least one other depot within connectivity range.
#     # If not, try to add a candidate from its connectivity coverage.
#     final_depots = set(selected)
#     for depot in selected:
#         # Check connectivity using full battery capacity radius
#         neighbors = {d for d in selected if d != depot and d in coverage_connect[depot]}
#         if not neighbors:
#             # Look for a candidate in connectivity coverage that connects to an existing depot
#             for candidate in coverage_connect[depot]:
#                 if candidate not in final_depots:
#                     candidate_neighbors = {d for d in final_depots if d in coverage_connect[candidate]}
#                     if candidate_neighbors:
#                         final_depots.add(candidate)
#                         break
#     return final_depots

def select_depots(G, battery_capacity, speed):
    """
    Greedy heuristic to select depot nodes.
    
    Parameters:
    - G: A networkx graph (must be weighted with 'travel_time' edges).
    - battery_capacity: Total allowed travel time (minutes) per trip.
    - speed: vehicle speed (consistent with edge length units) to compute travel time.
    
    Returns:
    - selected: set of depot nodes.
    """
    # Convert edge lengths to travel time
    G = convert_edge_weights_to_time(G, speed)
    
    # Maximum one-way travel time allowed (round-trip constraint)
    radius = battery_capacity / 2.0
    
    # Compute coverage for every node
    coverage = compute_coverage(G, radius)
    
    # Greedy selection
    all_nodes = set(G.nodes())
    selected = set()
    covered = set()
    
    while covered != all_nodes:
        # Select the node which adds the most uncovered nodes
        best_node = None
        best_gain = -1
        for node in G.nodes():
            if node in selected:
                continue
            gain = len(coverage[node] - covered)
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is None:
            break  # safety break in case no progress can be made
        selected.add(best_node)
        covered |= coverage[best_node]
    
    # Ensure each depot is reachable from at least one other depot
    # For any depot that is isolated, try to add another depot within its radius.
    final_depots = set(selected)
    for depot in selected:
        # Check if there exists another depot within radius (using precomputed coverage)
        neighbors = {d for d in selected if d != depot and d in coverage[depot]}#compute_coverage(G, 2*radius)}#coverage[depot]}
        if not neighbors:
            # Find a candidate (from coverage) that is not already selected to help connect this depot
            for candidate in coverage[depot]:
                # Only add if candidate improves connectivity with other depots
                candidate_neighbors = {d for d in selected if d in coverage[candidate]}
                if candidate_neighbors:
                    final_depots.add(candidate)
                    break
    return final_depots

def plot_graph_with_depots(G, depots):
    """
    Plot the graph G highlighting depot nodes in red.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw full graph (using x,y attributes)
    node_positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    # define node_colors, with nodes colors as green and depot nodes as red
    node_colors = ['green' if node not in depots else 'red' for node in G.nodes()]
    print(f'node_colors: {node_colors}')
    nx.draw_networkx_edges(G, pos=node_positions, ax=ax, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos=node_positions, node_size=10, node_color=node_colors, ax=ax)
    
    
    plt.title("Graph with Depots (red)")
    plt.axis('equal')
    plt.show()

def assign_node_colors(G, coverage_specific):
    # Create a base color list with a default color for all nodes
    node_colors = ['lightgray'] * len(G.nodes())
    
    # Convert nodes to list to ensure consistent indexing
    nodes_list = list(G.nodes())
    
    # Create a color map for different coverage groups
    color_map = list(mcolors.TABLEAU_COLORS.values())
    print(f'color_map: {len(color_map)}')
    # Assign colors to nodes in coverage_specific
    for i, (node, covered_nodes) in enumerate(coverage_specific.items()):
        # Choose a color from the color map (cycling if needed)
        color = color_map[i % len(color_map)]
        
        # Color the specific node and its covered nodes
        node_index = nodes_list.index(node)
        node_colors[node_index] = color
        
        for covered_node in covered_nodes:
            if covered_node in nodes_list:
                covered_index = nodes_list.index(covered_node)
                node_colors[covered_index] = color
    
    return node_colors

def remove_close_nodes(G, min_distance):
    # Get node coordinates using a different method
    def get_node_coordinates(G):
        return {node: (data['y'], data['x']) for node, data in G.nodes(data=True)}
    
    # Calculate great circle distance between two coordinate pairs
    def great_circle_distance(coord1, coord2):
        # Convert coordinates to radians
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in meters
        R = 6371000
        return c * R
    
    # Get node coordinates
    node_coords = get_node_coordinates(G)
    
    # Find nodes to remove
    nodes_to_remove = set()
    for (n1, n2) in combinations(G.nodes(), 2):
        dist = great_circle_distance(node_coords[n1], node_coords[n2])
        if dist < min_distance:
            # Choose the node with fewer edges to remove
            if G.degree(n1) <= G.degree(n2):
                nodes_to_remove.add(n1)
            else:
                nodes_to_remove.add(n2)
    
    # Remove nodes while preserving connectivity
    G_copy = G.copy()
    for node in nodes_to_remove:
        if node in G_copy:
            neighbors = list(G_copy.neighbors(node))
            G_copy.remove_node(node)
            # Add edges between neighbors to maintain connectivity
            for n1, n2 in combinations(neighbors, 2):
                if not G_copy.has_edge(n1, n2):
                    G_copy.add_edge(n1, n2)
    
    return G_copy

# Example usage:
if __name__ == '__main__':
    # Provided function to download and save graph from real-world roadmap
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
        # print(G.nodes(data=True))
        # fig, ax = ox.plot_graph(G, node_size=30, node_color='white', edge_linewidth=0.5)
        # print(G)

        # G_pruned = remove_close_nodes(G, min_distance=1000)  # 100 meters minimum distance
        # fig, ax = ox.plot_graph(G_pruned, node_size=30, node_color='white', edge_linewidth=0.5)
        # print(G)

        G = ox.consolidate_intersections(ox.project_graph(G.copy()), tolerance=threshold, rebuild_graph=True, dead_ends=False)
        fig, ax = ox.plot_graph(G, node_size=30, node_color='white', edge_linewidth=0.5)
        print(G)
        # G = ox.elevation.add_node_elevations_google(G, api_key=GOOGLE_MAPS_API_KEY)

        G = ox.project_graph(G, to_crs="EPSG:4326")
        
        return G

    # Define area and parameters
    # area_of_interest = "College Park, Maryland, US"
    # north, east, south, west = 38.99524, -76.92288, 38.98260, -76.94831
    # graph_name = "college_park.pickle"
    # G = download_and_save_graph(area_of_interest, north, east, south, west, graph_name)

    threshold = 300
    aoI = "Buffalo, New York, US"
    place = "new_york"
    north, east, south, west = 43.0669, -78.4644, 42.7873, -78.9800
    graph_name = "graph_files/" + place + ".pickle"
    G = download_and_save_graph(aoI, north, east, south, west, threshold, graph_name)

    # Parameters for depot selection
    battery_capacity = 40.0  # in minutes
    vehicle_speed = 480.0    # for example, 480 m/min (approx 28.8 km/h) - adjust as needed
    
    depots = select_depots(G, battery_capacity, vehicle_speed)

    coverage_specific = compute_coverage_specific_nodes(G, depots, battery_capacity / 2.0)
    print("Num selected depots:", len(depots))

    # Assign node colors
    node_colors = assign_node_colors(G, coverage_specific)
    # print(f'node_colors: {node_colors}')

    fig, ax = ox.plot_graph(G, node_size=30, node_color=node_colors, edge_linewidth=0.5)
    
    # plot_graph_with_depots(G, depots)
    node_colors = ['white' if node not in depots else 'r' for node in G.nodes()]

    fig, ax = ox.plot_graph(G, node_size=30, node_color=node_colors, edge_linewidth=0.5)
    

    plt.show()
