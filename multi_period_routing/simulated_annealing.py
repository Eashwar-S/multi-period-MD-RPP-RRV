"""
Contains function to implement simulated annealing metaheuristic
to improve the solutions produced by multi-trip algorithm.
"""

import math
from os import remove
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import tqdm


def convert_path_to_edges(paths):
    edges = []
    for i in range(len(paths) - 1):
        if paths[i] != paths[i + 1]:
            edges.append([paths[i], paths[i + 1]])
        else:
            raise Exception(f"Invalid edge - [{paths[i], paths[i+1]}]")
    return edges

def objective_function_vehicle_failures(G, solution, recharge_time, uavArrivalTimes):
    
    max_trip_time = -1
    for route_index, vehicle_route in enumerate(solution):
        vehicle_routing_time = 0
        if vehicle_route != 0:
            for trip_index, trips in enumerate(vehicle_route):
                for i in range(len(trips) - 1):
                    # print(trips[i], trips[i+1], trips)
                    try:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])[0]["weight"]
                    except:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])["weight"]
                if trip_index != len(vehicle_route) - 1:
                    vehicle_routing_time += recharge_time
        vehicle_routing_time += uavArrivalTimes[route_index]
        if vehicle_routing_time >= max_trip_time:
            max_trip_time = vehicle_routing_time
    return max_trip_time


def objective_function(G, solution, recharge_time):
    """
    Objective function is the maximum trip time of the solution.
    Args:
      G: A networkx weighted undirected multi-graph.
      solution: A list of vehicle routes. Each route is list of trip.
                Each trip is a list of sequence of nodes in the graph.
      recharge_time: A float specifying the time the vehicle spend
                     recharging after completing a trip.

    Returns:
      A float which is the maximum time vehicle spent routing.
    """
    max_trip_time = -1
    for vehicle_route in solution:
        vehicle_routing_time = 0
        if vehicle_route != 0:
            for trip_index, trips in enumerate(vehicle_route):
                for i in range(len(trips) - 1):
                    # print(trips[i], trips[i+1], trips)
                    try:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])[0]["weight"]
                    except:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])["weight"]
                if trip_index != len(vehicle_route) - 1:
                    vehicle_routing_time += recharge_time
        if vehicle_routing_time >= max_trip_time:
            max_trip_time = vehicle_routing_time
    return max_trip_time


def trip_times(G, solution, recharge_time):
    """
    Objective function is the maximum trip time of the solution.
    Args:
      G: A networkx weighted undirected multi-graph.
      solution: A list of vehicle routes. Each route is list of trip.
                Each trip is a list of sequence of nodes in the graph.
      recharge_time: A float specifying the time the vehicle spend
                     recharging after completing a trip.

    Returns:
      A float which is the maximum time vehicle spent routing.
    """
    vehicle_trip_times =[]
    vehicle_max_available_times = [0]*len(solution)
    max_trip_time = -1
    for vehicle_index, vehicle_route in enumerate(solution):
        vehicle_routing_time = 0
        if vehicle_route != 0 and vehicle_route:
            for trip_index, trips in enumerate(vehicle_route):
                trip_time = 0
                for i in range(len(trips) - 1):
                    try:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])[0]["weight"]
                        trip_time += G.get_edge_data(trips[i], trips[i + 1])[0]["weight"]
                    except:
                        vehicle_routing_time += G.get_edge_data(trips[i], trips[i + 1])["weight"]
                        trip_time += G.get_edge_data(trips[i], trips[i + 1])["weight"]
                
                if trip_index == 0: vehicle_trip_times.append([trip_time])
                else: vehicle_trip_times[-1].append(trip_time)
                
                if trip_index != len(vehicle_route) - 1:
                    vehicle_routing_time += recharge_time

            vehicle_max_available_times[vehicle_index] = vehicle_routing_time
        else:
            vehicle_trip_times.append([])
    
    return vehicle_max_available_times, vehicle_trip_times


def insertion(G, solution, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots):
    """
    This removes a trip which is a cycle from a vehicle routes and
    inserts it another vehicle routes.
    Args:
      solution: A list of vehicle routes. Each route is list of trip.
                Each trip is a list of sequence of nodes in the graph.

    Returns:
      A new feasible solution after the insertion.
    """
    if len(solution) > 1:
        # Rnadomly choosing two vehicle trips
        while True:
            insertion_vehicle_index, removal_vehicle_index = random.sample(range(0, len(solution)), 2)
            if len(solution[insertion_vehicle_index]) == 0 or len(solution[removal_vehicle_index]) == 0:
                insertion_vehicle_index, removal_vehicle_index = random.sample(range(0, len(solution)), 2)
            else:
                break
        v1_trip_index = random.randint(0, len(solution[insertion_vehicle_index])-1)
        v2_trip_index = random.randint(0, len(solution[removal_vehicle_index])-1)
        
        v2_trip = solution[removal_vehicle_index][v2_trip_index]
        
        # Get vehicle starting depots
        # vehicle_starting_depots = {i : solution[i][0][0] for i in range(len(solution))}

        insertion_results = {}

        # Make a copy of the soltion routes so that we don't modify the actual solution
        new_solution = copy.deepcopy(solution)
        # Insert vehicle trip
        new_solution[insertion_vehicle_index].insert(v1_trip_index, v2_trip)
        new_solution[removal_vehicle_index].remove(v2_trip)       
        new_solution = find_continuation_trip(G, new_solution, insertion_vehicle_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        insertion_results[obj] = new_solution

        new_solution = copy.deepcopy(solution)
        # Insert vehicle trip
        new_solution[insertion_vehicle_index].insert(v1_trip_index, v2_trip[::-1])
        new_solution[removal_vehicle_index].remove(v2_trip)       
        new_solution = find_continuation_trip(G, new_solution, insertion_vehicle_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        insertion_results[obj] = new_solution
        
        # print(swapping_results)
        return insertion_results[min(list(insertion_results.keys()))]
        # return insertion_results[list(insertion_results.keys())[random.randint(0, len(insertion_results)-1)]]
        
    return solution
    # new_solution = copy.deepcopy(solution)

    # locate_cycle_trips = []
    # # print('here')
    # for vehicle, vehicle_route in enumerate(new_solution):
    #     for trip_index, trips in enumerate(vehicle_route):
    #         if trips[0] == trips[-1]:
    #             locate_cycle_trips.append([vehicle, trip_index, trips[0]])
    #             # print(vehicle, trip_index, trips)
    
    # for vehicle, trip_index, node in locate_cycle_trips:
    #     other_vehicle = -1
    #     other_vehicle_list = []
    #     while len(other_vehicle_list) < len(new_solution) - 1:
    #         while True:
    #             other_vehicle = random.randint(0, len(new_solution)-1)
    #             if other_vehicle != vehicle and other_vehicle not in other_vehicle_list:
    #                 other_vehicle_list.append(other_vehicle)
    #                 break
    #         if len(new_solution[other_vehicle]) == 0:
    #                 new_solution[other_vehicle] = [new_solution[vehicle][trip_index]]
    #                 new_solution[vehicle].remove(new_solution[vehicle][trip_index])
    #                 return new_solution
                    
    #         else:
    #             for other_vehicle_trip_index, trips in enumerate(new_solution[other_vehicle]):
    #                 if trips[0] == node:
    #                     # print(trips, new_solution[other_vehicle], new_solution[vehicle], other_vehicle_trip_index, trip_index)
    #                     new_solution[other_vehicle].insert(other_vehicle_trip_index, new_solution[vehicle][trip_index])
    #                     new_solution[vehicle].remove(new_solution[vehicle][trip_index])
    #                     return new_solution
                        
    #                 elif other_vehicle_trip_index == len(new_solution[other_vehicle]) - 1:
    #                     if trips[-1] == node:
    #                         new_solution[other_vehicle].append(new_solution[vehicle][trip_index])
    #                         new_solution[vehicle].remove(new_solution[vehicle][trip_index])
    #                         return new_solution
                
    # return new_solution

def discontinued_trips(G, start, end, depot_nodes, vehicle_capacity):
    l = []
    while True:
        if start == end:
            return l
        else:    
            sub_G = nx.ego_graph(G, n=start, radius=vehicle_capacity, undirected=True, distance='weight')
            if end in list(sub_G.nodes()):
                l.append(list(nx.astar_path(sub_G, source=start, target=end)))
                return l
            else:   
                minCost = np.inf
                nearestDepotNode = -1
                for node in depot_nodes:
                    if node in list(sub_G.nodes()):
                        if node != start:
                            dis1 = nx.astar_path_length(sub_G, source=start, target=node)
                            dis2 = nx.astar_path_length(G, source=node, target=end)
                            if dis1 + dis2 < minCost:
                                minCost = dis1 + dis2
                                l.append(list(nx.astar_path(sub_G, source=start, target=node)))
                                nearestDepotNode = node
                if nearestDepotNode != -1:
                    start = nearestDepotNode
                else:
                    print('Something is wrong')
                    break
    return l

def find_continuation_trip(G, vehicle_routes, vehicle_index, depot_nodes, vehicle_capacity, vehicle_starting_depots):
    """
    check if vehicles trips inside each vehicle routes end and begin at the same node
    """
    while True:
        trip_index = -1
        if vehicle_routes[vehicle_index][0][0] != vehicle_starting_depots[vehicle_index]:
            start, end = vehicle_starting_depots[vehicle_index], vehicle_routes[vehicle_index][0][0]
            trip_index = 0
        elif len(vehicle_routes[vehicle_index]) != 1:
            for i in range(len(vehicle_routes[vehicle_index])-1):
                if vehicle_routes[vehicle_index][i][-1] != vehicle_routes[vehicle_index][i+1][0]:
                    start, end = vehicle_routes[vehicle_index][i][-1], vehicle_routes[vehicle_index][i+1][0]           
                    trip_index = i+1
                    break
        if trip_index == -1:
            break
        trips = discontinued_trips(G, start, end, depot_nodes, vehicle_capacity)
        for trip in trips:
            if len(trip) != 0:
                vehicle_routes[vehicle_index].insert(trip_index, trip)
                trip_index += 1
        
    return vehicle_routes
        


def swap(G, solution, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots):
    """
    This removes the last trip from a vehicle routes and
    swaps it with other vehicle's routes lasst trip if feasible
    swapping is possible.
    Args:
      solution: A list of vehicle routes. Each route is list of trip.
                Each trip is a list of sequence of nodes in the graph.
      recharge_time: A float specifying the time the vehicle spend
                     recharging after completing a trip.

    Returns:
      A new feasible solution after the insertion.
    """
    # print('before swapping')
    # print(solution)
    if len(solution) > 1:
        # Rnadomly choosing two vehicle trips
        while True:
            v1_index, v2_index = random.sample(range(0, len(solution)), 2)
            if len(solution[v1_index]) == 0 or len(solution[v2_index]) == 0:
                v1_index, v2_index = random.sample(range(0, len(solution)), 2)
            else:
                break
        
        v1_trip_index = random.randint(0, len(solution[v1_index])-1)
        v2_trip_index = random.randint(0, len(solution[v2_index])-1)
        # print('trip indexes')
        # print(v1_trip_index, v2_trip_index)
        v1_trip, v2_trip = solution[v1_index][v1_trip_index], solution[v2_index][v2_trip_index]
        
        # Get vehicle starting depots
        # vehicle_starting_depots = {i : solution[i][0][0] for i in range(len(solution))}

        # Four possible way to swap two trips keeping track of all possible swapping results
        # chossing the best reults.
        swapping_results = {}

        # Make a copy of the soltion routes so that we don't modify the actual solution
        new_solution = copy.deepcopy(solution)
        # Swap vehicle trips
        new_solution[v1_index][v1_trip_index], new_solution[v2_index][v2_trip_index] = v2_trip, v1_trip       
        new_solution = find_continuation_trip(G, new_solution, v1_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        new_solution = find_continuation_trip(G, new_solution, v2_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        swapping_results[obj] = new_solution

        new_solution = copy.deepcopy(solution)
        # Swap vehicle trips
        new_solution[v1_index][v1_trip_index], new_solution[v2_index][v2_trip_index] = v2_trip[::-1], v1_trip
        new_solution = find_continuation_trip(G, new_solution, v1_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        new_solution = find_continuation_trip(G, new_solution, v2_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        swapping_results[obj] = new_solution

        new_solution = copy.deepcopy(solution)
        # Swap vehicle trips
        new_solution[v1_index][v1_trip_index], new_solution[v2_index][v2_trip_index] = v2_trip, v1_trip[::-1]
        new_solution = find_continuation_trip(G, new_solution, v1_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        new_solution = find_continuation_trip(G, new_solution, v2_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        swapping_results[obj] = new_solution

        new_solution = copy.deepcopy(solution)
        # Swap vehicle trips
        new_solution[v1_index][v1_trip_index], new_solution[v2_index][v2_trip_index] = v2_trip[::-1], v1_trip[::-1]
        new_solution = find_continuation_trip(G, new_solution, v1_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        new_solution = find_continuation_trip(G, new_solution, v2_index, depotNodes, vehicleCapacity, vehicle_starting_depots)
        obj = objective_function(G, new_solution, recharge_time)
        swapping_results[obj] = new_solution
        
        # print(swapping_results)
        return swapping_results[min(list(swapping_results.keys()))]
        # return swapping_results[list(swapping_results.keys())[random.randint(0, len(swapping_results)-1)]]
        
    return solution



    # swap_flag = False
    # combinations = []
    # K = len(solution)                           # Number of Vehicles 
    # while len(combinations) < K*(K-1)//2:       # Randomly choose two different vehicles
    #     while True:
    #         vehicle_1 = random.randint(0, len(solution)-1)
    #         vehicle_2 = random.randint(0, len(solution)-1)
    #         if vehicle_2 != vehicle_1 and [vehicle_1, vehicle_2] not in combinations and [vehicle_2, vehicle_1] not in combinations:
    #             combinations.append([vehicle_1, vehicle_2])
    #             break

    #     if len(new_solution[vehicle_1]) == 0:
    #         if len(new_solution[vehicle_2]) == 0:
    #             continue
    #         else:
    #             new_solution[vehicle_1] = [new_solution[vehicle_2][-1]]
    #             new_solution[vehicle_2].remove(new_solution[vehicle_2][-1])
    #             swap_flag = True
    #     else:
    #         if len(new_solution[vehicle_2]) == 0:
    #             new_solution[vehicle_2] = [new_solution[vehicle_1][-1]]
    #             new_solution[vehicle_1].remove(new_solution[vehicle_1][-1])
    #             swap_flag = True
    #         else:
    #             if new_solution[vehicle_1][-1][0] == new_solution[vehicle_2][-1][0]:
    #                 v1_trip = new_solution[vehicle_1].pop()
    #                 v2_trip = new_solution[vehicle_2].pop()
    #                 new_solution[vehicle_1].append(v2_trip)
    #                 new_solution[vehicle_2].append(v1_trip)
    #                 swap_flag = True
    #             elif new_solution[vehicle_1][-1][0] == new_solution[vehicle_2][-1][-1] and new_solution[vehicle_1][-1][-1] == new_solution[vehicle_2][-1][0]:
    #                 v1_trip = new_solution[vehicle_1].pop()
    #                 v2_trip = new_solution[vehicle_2].pop()
    #                 new_solution[vehicle_1].append(v2_trip[::-1])
    #                 new_solution[vehicle_2].append(v1_trip[::-1])
    #                 swap_flag = True
    #     if swap_flag:
    #         break
    # return new_solution

def nearestDepot(G, node, depotNodes, radius):
    time = np.inf
    path = []
    dic = {}
    for nodes in depotNodes:
        dic[nodes] = 'yes'
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=False, distance='weight')
    
    for nodes in subG.nodes():
        if dic.get(nodes) != None: 
            t = nx.astar_path_length(G, node, nodes)
            if time > t:
                time = t
                path = nx.astar_path(G, node, nodes)
    return time, path


def feasibleTraversalOfEdge(G, node, edges):
    if node in edges:
        if node == edges[0]:
            return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[1]]
        else:
            return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[0]]
    time = np.inf
    path = []
    l1 = nx.astar_path(G, source=node, target=edges[0])
    l2 = nx.astar_path(G, source=node, target=edges[1])

    c1 = nx.astar_path_length(G, source=node, target=edges[0])
    c2 = nx.astar_path_length(G, source=node, target=edges[1])

    if edges[1] != l1[-2]:
        c1 += G.get_edge_data(edges[0], edges[1], 0)['weight']
        l1.append(edges[1])

    if edges[0] != l2[-2]:
        c2 += G.get_edge_data(edges[0], edges[1], 0)['weight']
        l2.append(edges[0])
    # print(l1, l2, c1, c2)
    if c1 <= c2:
        time = c1
        path = l1
    else:
        time = c2
        path = l2
    return time, path


def feasibleTraversalOfEdgeModified(G, node, edges, depotNodes, vehicleCapacity):
    time = np.inf
    path = []
    
    if node in edges:
        if node == edges[0]:
            l1 = [edges[0], edges[1]]
            c1 = G.get_edge_data(edges[0], edges[1], 0)['weight']
            l2 = [edges[0], edges[1], edges[0]]
            c2 = 2*c1
            # return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[1]]
        else:
            l2 = [edges[1], edges[0]]
            c2 = G.get_edge_data(edges[0], edges[1], 0)['weight']
            l1 = [edges[1], edges[0], edges[1]]
            c1 = 2*c2
            # return G.get_edge_data(edges[0], edges[1], 0)['weight'], [node, edges[0]]
    else:
        l1 = nx.astar_path(G, source=node, target=edges[0])
        l2 = nx.astar_path(G, source=node, target=edges[1])

        c1 = nx.astar_path_length(G, source=node, target=edges[0])
        c2 = nx.astar_path_length(G, source=node, target=edges[1])

        if edges[1] != l1[-2]:
            c1 += G.get_edge_data(edges[0], edges[1], 0)['weight']
            l1.append(edges[1])

        if edges[0] != l2[-2]:
            c2 += G.get_edge_data(edges[0], edges[1], 0)['weight']
            l2.append(edges[0])
    # print(l1, l2, c1, c2)

    c1_d, t1 = nearestDepot(G, l1[-1], depotNodes, vehicleCapacity-c1)
    c2_d, t2 = nearestDepot(G, l2[-1], depotNodes, vehicleCapacity-c2)
    
    # print(f'return - times - {c1_d, c2_d}, paths - {t1, t2}')
    if c1 + c1_d <= vehicleCapacity:
        if c2 + c2_d <= vehicleCapacity:
            if c1 + c1_d <= c2 + c2_d:
                time = c1
                path = l1
            else:
                time = c2
                path = l2
        else:
            time = c1
            path = l1
    elif c2 + c2_d <= vehicleCapacity:
        time = c2
        path = l2
    # print(f'feasibility - time - {time}, path - {path}')
    return time, path


def bestFlight(G, node, Te, untraversedEdges, depotNodes, radius, threshold):
    # print(node, Te, untraversedEdges, depotNodes, radius, threshold)
    dic = {}
    pathTime = np.inf
    path = []
    Tstar = np.inf
    Estar = 0
    Qstar = []
    nearestEdgeIndex = -1
    for edges in untraversedEdges:
        dic[str(tuple(edges))] = 'yes'
        
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=True, distance='weight')

    for edges in subG.edges():
        if dic.get(str(edges)) != None or dic.get(str(edges[::-1])) != None:

            if dic.get(str(edges)) != None:
                edgeIndex = untraversedEdges.index(list(edges))
            elif dic.get(str(edges[::-1])) != None:
                edgeIndex = untraversedEdges.index(list(edges[::-1]))

            # Te[edgeIndex], Q1 = feasibleTraversalOfEdge(G, node, edges)
            Te[edgeIndex], Q1 = feasibleTraversalOfEdgeModified(G, node, edges, depotNodes, radius)
            # print('Eashwar')
            # print(Te[edgeIndex], Q1, node, edges)
            timeQ2, Q2 = nearestDepot(G, Q1[-1], depotNodes, radius - Te[edgeIndex])#radius - Te[edgeIndex]

            if Te[edgeIndex] + timeQ2 <= threshold and Te[edgeIndex] < Tstar:
                Tstar = Te[edgeIndex]
                Estar = untraversedEdges[edgeIndex]
                Qstar = Q1
                nearestEdgeIndex = edgeIndex
#             print(Tstar)
    # print(Estar, Tstar, Qstar, nearestEdgeIndex, Te)     
    return Estar, Tstar, Qstar, nearestEdgeIndex, Te


def bestMultiFlight(G, node, Vd, Td, depotNodes, untraversedEdges, radius, threshold, Te):
    dic = {}
    nearestEdge = np.argmin(Te)
    for nodes in depotNodes:
        dic[nodes] = 'yes'
    
    subG = nx.ego_graph(G, n=node, radius=radius, undirected=False, distance='weight')
    
    for nodes in subG.nodes():
        if dic.get(nodes) != None:
            depotIndex = depotNodes.index(nodes)
            Q1 = nx.astar_path(G, source=node, target=nodes)
            Td[depotIndex] = nx.astar_path_length(G, source=node, target=nodes)
            Te_, Q2 = feasibleTraversalOfEdgeModified(G, nodes, untraversedEdges[nearestEdge], depotNodes, radius)
            
            Vd[depotIndex] = Td[depotIndex] + Te_
            
            if Td[depotIndex] > threshold or node == nodes:
                    Vd[depotIndex] = np.inf
    
    return Vd

def remove_traversed_edges(untraversedEdges, edges_traversed, path):
    e_list = convert_path_to_edges(path)
    for edge in untraversedEdges:
        if edge in e_list or edge[::-1] in e_list:
            edges_traversed.append(edge)

    for edge in edges_traversed:
        if edge in untraversedEdges:
            untraversedEdges.remove(edge)
    return untraversedEdges, edges_traversed

def single_trip(G, uav_loc, untraversedEdges, depotNodes, edges_traversed, vehicleCapacity):
    trip = []
    trip_time = 0
    while True:

        if untraversedEdges and trip_time < vehicleCapacity:
            Tstar, Qstar = feasibleTraversalOfEdgeModified(G, uav_loc, untraversedEdges[0], depotNodes, vehicleCapacity - trip_time)
        else:
            if trip and trip[-1] not in depotNodes:
                Tstar = np.inf
            else:
                break
        if Tstar != np.inf and Tstar + trip_time <= vehicleCapacity:
            # Check for traversed required edges
            untraversedEdges, edges_traversed = remove_traversed_edges(untraversedEdges, edges_traversed, copy.deepcopy(Qstar))
    
            # Update trip
            if not trip:
                trip = Qstar
            else:
                trip += Qstar[1:]
            trip_time += Tstar
            
            uav_loc = Qstar[-1]
        else:
            if trip: 
                timeQ1, Q1 = nearestDepot(G, trip[-1], depotNodes, vehicleCapacity - trip_time)
                if timeQ1 != np.inf:
                    trip_time += timeQ1
                    trip += Q1[1:]
                    untraversedEdges, edges_traversed = remove_traversed_edges(untraversedEdges, edges_traversed, copy.deepcopy(Q1))
            break
        # print(f'trip - {trip}, trip_time - {trip_time}, vehicle cap - {vehicleCapacity}')
    return trip, trip_time, untraversedEdges, edges_traversed
    
def move_to_closer_depot(G, uav_loc, untraversedEdges, depotNodes, vehicleCapacity):
    Estar = untraversedEdges[0]
    Vd = np.array([np.inf]*len(depotNodes), dtype=np.float32)

    for depotIndex, node in enumerate(depotNodes):
        if node != uav_loc:
            # Q1 = nx.astar_path(G, source=uav_loc, target=node)
            # T1 = nx.astar_path_length(G, source=uav_loc, target=node)
            # if T1 <= vehicleCapacity:
            T2, Q2 = feasibleTraversalOfEdgeModified(G, node, Estar, depotNodes, vehicleCapacity)                    
            # print(f'T2 Q2 - {T2, Q2}')
            if T2 <= vehicleCapacity:
                Q1 = nx.astar_path(G, source=uav_loc, target=node)
                T1 = nx.astar_path_length(G, source=uav_loc, target=node)
                # need to insert an vehiclecapacity condition here
                Vd[depotIndex] = T1 + T2
            else:
                Vd[depotIndex] = np.inf
        else:
            Vd[depotIndex] = np.inf
    
    chosen_depot_node = depotNodes[np.argmin(Vd)]
    path = nx.astar_path(G, source=uav_loc, target=chosen_depot_node)
    path_time = nx.astar_path_length(G, source=uav_loc, target=chosen_depot_node)
    untraversedEdges, edges_traversed = remove_traversed_edges(untraversedEdges, [], copy.deepcopy(path))

    return path, path_time, untraversedEdges, edges_traversed

def routes_based_on_allocation(G, untraversedEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                  uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity):

    multiFlight = False
    numRecharges = 0
    edges_traversed = []
    uav_to_requirededges = {}
    start = time.time()
    threshold = len(untraversedEdges)
    pre = len(untraversedEdges)
    # print(G.edges())
    while untraversedEdges:
        # print(untraversedEdges, uavPaths)
        uav = 0#np.argmin(uavAvailableTime)
        # Te = np.array([np.inf], dtype=np.float32)

        # Estar, Tstar, Qstar, nearestEdgeIndex, Te = bestFlight(G, uavLocation[uav], Te, [untraversedEdges[0]], depotNodes, vehicleCapacity - uavUtilization[uav], vehicleCapacity - uavUtilization[uav])
        # print('start')
        # print(untraversedEdges, uavLocation[uav], vehicleCapacity, depotNodes)
        
        # print('Single trip testing')

        tr, tr_t, untraversedEdges, edges_traversed = single_trip(G, uavLocation[uav], untraversedEdges, depotNodes, edges_traversed, vehicleCapacity)
        # print(tr, tr_t, untraversedEdges, edges_traversed, uav_to_requirededges)

        if tr_t != 0:
            if uavPaths[uav] == 0:
                uavPaths[uav] = [tr]
                uavPathTimes[uav] = [tr_t]
            else:
                uavPaths[uav].append(tr)
                uavPathTimes[uav].append(tr_t)
            pass
            uavLocation[uav] = tr[-1]
            if uav not in uav_to_requirededges:
                uav_to_requirededges[uav] = copy.deepcopy(edges_traversed)
            else:
                uav_to_requirededges[uav] += copy.deepcopy(edges_traversed)
            edges_traversed.clear()
        else:
            path, path_time, untraversedEdges, edges_traversed = move_to_closer_depot(G, uavLocation[uav], untraversedEdges, depotNodes, vehicleCapacity)
            if uavPaths[uav] == 0:
                uavPaths[uav] = [path]
                uavPathTimes[uav] = [path_time]
            else:
                uavPaths[uav].append(path)
                uavPathTimes[uav].append(path_time)
            pass
            uavLocation[uav] = path[-1]
            if uav not in uav_to_requirededges:
                uav_to_requirededges[uav] = copy.deepcopy(edges_traversed)
            else:
                uav_to_requirededges[uav] += copy.deepcopy(edges_traversed)
            edges_traversed.clear()

    uavLastArrivalTimes = [0]
    try:
        uavLastArrivalTimes[0] = sum(uavPathTimes[0]) + max(0, len(uavPathTimes[0])-1)*rechargeTime
    except:
        uavLastArrivalTimes = [0]
    # print('check results')
    # print(uavPaths)
    # print(uavPathTimes)
    # print(uavLastArrivalTimes)
    # print(uav_to_requirededges)
    return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges


def routes_based_on_allocation_ts(G, untraversedEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                  uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity):

    multiFlight = False
    numRecharges = 0
    edges_traversed = []
    uav_to_requirededges = {}
    start = time.time()
    threshold = len(untraversedEdges)
    pre = len(untraversedEdges)
    start = time.time()
    valid_allocation = True
    # print(G.edges())
    while untraversedEdges:
        # print(untraversedEdges, uavPaths)
        if pre - len(untraversedEdges) != 0:
            pre = len(untraversedEdges)
            start = time.time()
        
        if time.time() - start > 5:
            valid_allocation = False
            break

        uav = 0#np.argmin(uavAvailableTime)
        # Te = np.array([np.inf], dtype=np.float32)

        # Estar, Tstar, Qstar, nearestEdgeIndex, Te = bestFlight(G, uavLocation[uav], Te, [untraversedEdges[0]], depotNodes, vehicleCapacity - uavUtilization[uav], vehicleCapacity - uavUtilization[uav])
        # print('start')
        # print(untraversedEdges, uavLocation[uav], vehicleCapacity, depotNodes)
        
        # print('Single trip testing')

        tr, tr_t, untraversedEdges, edges_traversed = single_trip(G, uavLocation[uav], untraversedEdges, depotNodes, edges_traversed, vehicleCapacity)
        # print(tr, tr_t, untraversedEdges, edges_traversed, uav_to_requirededges)

        if tr_t != 0:
            if uavPaths[uav] == 0:
                uavPaths[uav] = [tr]
                uavPathTimes[uav] = [tr_t]
            else:
                uavPaths[uav].append(tr)
                uavPathTimes[uav].append(tr_t)
            pass
            uavLocation[uav] = tr[-1]
            if uav not in uav_to_requirededges:
                uav_to_requirededges[uav] = copy.deepcopy(edges_traversed)
            else:
                uav_to_requirededges[uav] += copy.deepcopy(edges_traversed)
            edges_traversed.clear()
        else:
            path, path_time, untraversedEdges, edges_traversed = move_to_closer_depot(G, uavLocation[uav], untraversedEdges, depotNodes, vehicleCapacity)
            if uavPaths[uav] == 0:
                uavPaths[uav] = [path]
                uavPathTimes[uav] = [path_time]
            else:
                uavPaths[uav].append(path)
                uavPathTimes[uav].append(path_time)
            pass
            uavLocation[uav] = path[-1]
            if uav not in uav_to_requirededges:
                uav_to_requirededges[uav] = copy.deepcopy(edges_traversed)
            else:
                uav_to_requirededges[uav] += copy.deepcopy(edges_traversed)
            edges_traversed.clear()

    uavLastArrivalTimes = [0]
    try:
        uavLastArrivalTimes[0] = sum(uavPathTimes[0]) + max(0, len(uavPathTimes[0])-1)*rechargeTime
    except:
        uavLastArrivalTimes = [0]
    # print('check results')
    # print(uavPaths)
    # print(uavPathTimes)
    # print(uavLastArrivalTimes)
    # print(uav_to_requirededges)
    return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges, valid_allocation


def routes_based_on_allocation_temp(G, untraversedEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                  uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity):

    multiFlight = False
    numRecharges = 0
    count = 0
    uav_to_requirededges = {}
    start = time.time()
    threshold = len(untraversedEdges)
    pre = len(untraversedEdges)
    # print(G.edges())
    while untraversedEdges:
        # print(len(untraversedEdges))
        if count > len(depotNodes) + 5:
            return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges
        count += 1
        # print(untraversedEdges, uavPaths)
        # print(untraversedEdges, len(untraversedEdges))
        # count = 0
        uav = np.argmin(uavAvailableTime)
        # if uav == 2:s
        #     print(uavPaths)
        Te = np.array([np.inf]*len(untraversedEdges[0]), dtype=np.float32)
        Estar, Tstar, Qstar, nearestEdgeIndex, Te = bestFlight(G, uavLocation[uav], Te, [untraversedEdges[0]], depotNodes, vehicleCapacity - uavUtilization[uav], vehicleCapacity - uavUtilization[uav])
        # print(uavPaths, untraversedEdges[nearestEdgeIndex], Estar, uav, Qstar, Tstar)
        if Estar != 0:
            if uavPaths[uav] == 0:
                uavPaths[uav] = [Qstar]
                uavPathTimes[uav] = [Tstar]
            else:
                if multiFlight:
                    if uavLocation[uav] not in depotNodes:
                        uavPaths[uav][-1] += Qstar[1:]
                        uavPathTimes[uav][-1] += Tstar + rechargeTime
                    else:
                        uavPaths[uav].append(Qstar)
                        uavPathTimes[uav].append(Tstar)
                    numRecharges += 1
                elif not multiFlight and uavLocation[uav] in depotNodes:
                    uavPaths[uav].append(Qstar)
                    uavPathTimes[uav].append(Tstar)
                elif not multiFlight and uavLocation[uav] not in depotNodes:
                    uavPaths[uav][-1] += Qstar[1:]
                    uavPathTimes[uav][-1] += Tstar
            if uav_to_requirededges.get(uav) != None:
                    uav_to_requirededges[uav].append(untraversedEdges[nearestEdgeIndex])
            else:
                uav_to_requirededges[uav] = [untraversedEdges[nearestEdgeIndex]]
                    

            uavUtilization[uav] += Tstar
            uavAvailableTime[uav] += Tstar
            # print(multiFlight, uavPaths[uav], uavPaths[uav][-1][-1], Qstar)
            uavLocation[uav] = Qstar[-1]

            if uavLocation[uav] in depotNodes:
                uavLastArrivalTimes[uav] = uavAvailableTime[uav]
                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavUtilization[uav] = 0
                multiFlight = False

            untraversedEdges.remove(untraversedEdges[nearestEdgeIndex])

        else:
            Estar = untraversedEdges[np.argmin(Te)]
            Td = [np.inf]*len(depotNodes)
            Vd = np.array([np.inf]*len(depotNodes), dtype=np.float32)
            if uavLocation[uav] in depotNodes:
                multiFlight = True
            else:
                multiFlight = False
            
            
            Vd = bestMultiFlight(G, uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te) #vehicleCapacity - uavUtilization[uav]
            # print('Eashwar')
            # print(uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te)
            # print(Vd)
            # print(uav)
            nearestFeasibleDepot = depotNodes[np.argmin(Vd)]
            if Vd[np.argmin(Vd)] < np.inf:
                Q1 = nx.astar_path(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                TQ1 = nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                if uavPaths[uav] == 0:
                    uavPaths[uav] = [Q1]
                    uavPathTimes[uav] = [TQ1]
                else:
                    if not multiFlight:
                        uavPaths[uav][-1] += Q1[1:]
                        uavPathTimes[uav][-1] += TQ1
                    else:
                        uavPaths[uav].append(Q1)
                        uavPathTimes[uav].append(TQ1)
                uavLastArrivalTimes[uav] = uavAvailableTime[uav] + nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])#depotToDepotDistance[depotNodes.index(uavLocation[uav]), np.argmin(Vd)]

                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavLocation[uav] = nearestFeasibleDepot
                uavUtilization[uav] = 0
            else:
                uavAvailableTime[uav] = np.inf
    
    # print('Finished Designing Paths')
    # print('Completing Incomplete Paths')
    # print(uavPaths)
    # print(uavPathTimes)
    for k in range(totalUavs):
        if uavLocation[k] not in depotNodes:
            timeQ1, Q1 = nearestDepot(G, uavLocation[k], depotNodes, vehicleCapacity - uavUtilization[k])
            uavPaths[k][-1] += Q1[1:]
            uavPathTimes[k][-1] += timeQ1
            uavLastArrivalTimes[k] = uavAvailableTime[k] + timeQ1
            uavLocation[k] = Q1[-1]
        uavLastArrivalTimes[k] = round(uavLastArrivalTimes[k], 1)
        if uavPathTimes[k] != 0:
            for i in range(len(uavPathTimes[k])):
                uavPathTimes[k][i] = round(uavPathTimes[k][i], 1)
            # print(uavPaths)
    # untraversedEdges.clear()
    # print(uav_to_requirededges)
    return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges



def newHeuristics(G, untraversedEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                  uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity):

    multiFlight = False
    numRecharges = 0
    count = 0
    uav_to_requirededges = {}
    start = time.time()
    threshold = len(untraversedEdges)
    pre = len(untraversedEdges)
    # print(G.edges())
    while untraversedEdges:
        # print(len(untraversedEdges))
        if count > len(depotNodes) + 5:
            return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges
        count += 1
        # print(untraversedEdges, uavPaths)
        # print(untraversedEdges, len(untraversedEdges))
        # count = 0
        uav = np.argmin(uavAvailableTime)
        # if uav == 2:s
        #     print(uavPaths)
        Te = np.array([np.inf]*len(untraversedEdges), dtype=np.float32)
        Estar, Tstar, Qstar, nearestEdgeIndex, Te = bestFlight(G, uavLocation[uav], Te, untraversedEdges, depotNodes, vehicleCapacity - uavUtilization[uav], vehicleCapacity - uavUtilization[uav])
        # print(uavPaths, untraversedEdges[nearestEdgeIndex], Estar, uav, Qstar, Tstar)
        if Estar != 0:
            if uavPaths[uav] == 0:
                uavPaths[uav] = [Qstar]
                uavPathTimes[uav] = [Tstar]
            else:
                if multiFlight:
                    if uavLocation[uav] not in depotNodes:
                        uavPaths[uav][-1] += Qstar[1:]
                        uavPathTimes[uav][-1] += Tstar + rechargeTime
                    else:
                        uavPaths[uav].append(Qstar)
                        uavPathTimes[uav].append(Tstar)
                    numRecharges += 1
                elif not multiFlight and uavLocation[uav] in depotNodes:
                    uavPaths[uav].append(Qstar)
                    uavPathTimes[uav].append(Tstar)
                elif not multiFlight and uavLocation[uav] not in depotNodes:
                    uavPaths[uav][-1] += Qstar[1:]
                    uavPathTimes[uav][-1] += Tstar
            if uav_to_requirededges.get(uav) != None:
                    uav_to_requirededges[uav].append(untraversedEdges[nearestEdgeIndex])
            else:
                uav_to_requirededges[uav] = [untraversedEdges[nearestEdgeIndex]]
                    

            uavUtilization[uav] += Tstar
            uavAvailableTime[uav] += Tstar
            # print(multiFlight, uavPaths[uav], uavPaths[uav][-1][-1], Qstar)
            uavLocation[uav] = Qstar[-1]

            if uavLocation[uav] in depotNodes:
                uavLastArrivalTimes[uav] = uavAvailableTime[uav]
                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavUtilization[uav] = 0
                multiFlight = False

            untraversedEdges.remove(untraversedEdges[nearestEdgeIndex])

        else:
            Estar = untraversedEdges[np.argmin(Te)]
            Td = [np.inf]*len(depotNodes)
            Vd = np.array([np.inf]*len(depotNodes), dtype=np.float32)
            if uavLocation[uav] in depotNodes:
                multiFlight = True
            else:
                multiFlight = False
            
            
            Vd = bestMultiFlight(G, uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te) #vehicleCapacity - uavUtilization[uav]
            # print('Eashwar')
            # print(uavLocation[uav], Vd, Td, depotNodes, untraversedEdges, vehicleCapacity, vehicleCapacity - uavUtilization[uav],  Te)
            # print(Vd)
            # print(uav)
            nearestFeasibleDepot = depotNodes[np.argmin(Vd)]
            if Vd[np.argmin(Vd)] < np.inf:
                Q1 = nx.astar_path(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                TQ1 = nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])
                if uavPaths[uav] == 0:
                    uavPaths[uav] = [Q1]
                    uavPathTimes[uav] = [TQ1]
                else:
                    if not multiFlight:
                        uavPaths[uav][-1] += Q1[1:]
                        uavPathTimes[uav][-1] += TQ1
                    else:
                        uavPaths[uav].append(Q1)
                        uavPathTimes[uav].append(TQ1)
                uavLastArrivalTimes[uav] = uavAvailableTime[uav] + nx.astar_path_length(G, uavLocation[uav], depotNodes[np.argmin(Vd)])#depotToDepotDistance[depotNodes.index(uavLocation[uav]), np.argmin(Vd)]

                uavAvailableTime[uav] = uavLastArrivalTimes[uav] + rechargeTime
                numRecharges += 1
                uavLocation[uav] = nearestFeasibleDepot
                uavUtilization[uav] = 0
            else:
                uavAvailableTime[uav] = np.inf
    
    # print('Finished Designing Paths')
    # print('Completing Incomplete Paths')
    # print(uavPaths)
    # print(uavPathTimes)
    for k in range(totalUavs):
        if uavLocation[k] not in depotNodes:
            timeQ1, Q1 = nearestDepot(G, uavLocation[k], depotNodes, vehicleCapacity - uavUtilization[k])
            uavPaths[k][-1] += Q1[1:]
            uavPathTimes[k][-1] += timeQ1
            uavLastArrivalTimes[k] = uavAvailableTime[k] + timeQ1
            uavLocation[k] = Q1[-1]
        uavLastArrivalTimes[k] = round(uavLastArrivalTimes[k], 1)
        if uavPathTimes[k] != 0:
            for i in range(len(uavPathTimes[k])):
                uavPathTimes[k][i] = round(uavPathTimes[k][i], 1)
            # print(uavPaths)
    # untraversedEdges.clear()
    # print(uav_to_requirededges)
    return uavPaths, uavPathTimes, uavLastArrivalTimes, untraversedEdges, numRecharges, uav_to_requirededges


def generate_new_solution(G, solution, rechargeTime, depotNodes, vehicleCapacity, vehicle_starting_depots, uavToRequiredEdges, originalLocation, uavLastArrivalTimes):
    """
    Function to generate new solution by performing insertion and swapping
    operations.
    Args:
      solution: A list of vehicle routes. Each route is list of trip.
                Each trip is a list of sequence of nodes in the graph.

    Returns:
      A new feasible solution route.
    """
    copy_uavToRequiredEdges = copy.deepcopy(uavToRequiredEdges)
    # print(f"uavToRequiredEdges : {uavToRequiredEdges}")
    copy_uavLastArrivalTimes = np.copy(uavLastArrivalTimes)
    # print(solution, uavToRequiredEdges)
    # print(f"Before swapping or insertion mapping : {uavToRequiredEdges}")
    if len(solution) >= 2:
        u1 = -1
        u2 = -1
        if random.randint(1,10) >= 5 and len([edge for edge in list(uavToRequiredEdges.values()) if edge]) >= 2:
            # Randomly swapping required edges allocation
            # print('swapping')
            # print(uavToRequiredEdges)
            while True:
                uav_1 = random.randint(0, len(solution)-1)
                if uav_1 in list(uavToRequiredEdges.keys()) and len(uavToRequiredEdges[uav_1]) >= 1:
                    break
            while True:
                temp_ = random.randint(0, len(solution)-1)
                if temp_ != uav_1 and temp_ in list(uavToRequiredEdges.keys())  and len(uavToRequiredEdges[temp_]) >= 1:
                    uav_2 = temp_
                    break

            req_edge_ind_max_uav = random.randint(0, len(uavToRequiredEdges[uav_1])-1)
            req_edge_ind_min_uav = random.randint(0, len(uavToRequiredEdges[uav_2])-1)

            # Swapping
            temp = uavToRequiredEdges[uav_1][req_edge_ind_max_uav]
            uavToRequiredEdges[uav_1][req_edge_ind_max_uav] = uavToRequiredEdges[uav_2][req_edge_ind_min_uav]
            uavToRequiredEdges[uav_2][req_edge_ind_min_uav] = temp
            u1 = uav_1
            u2 = uav_2
            # print('here1')
            # print(uavToRequiredEdges)
        else:
            # print('insertion')
            # # Random insertion of max used trip 
            # print(uavLastArrivalTimes, np.argmax(uavLastArrivalTimes))
            # print(uavToRequiredEdges)
            # print(solution)
            vehicle = []
            while True:
                # max_used_vehicle = list(uavLastArrivalTimes).index(max([t for t in list(uavLastArrivalTimes) if t not in vehicle]))#np.argmax(uavLastArrivalTimes)
                max_used_vehicle = np.argmax(uavLastArrivalTimes)
                # print(max_used_vehicle, uavToRequiredEdges, uavLastArrivalTimes)
                if max_used_vehicle in list(uavToRequiredEdges.keys()):
                    break
                else:
                    uavLastArrivalTimes[max_used_vehicle] = 0
                #     vehicle.append(max_used_vehicle)
            # print('here1')
            while True:
                temp_ = random.randint(0, len(solution)-1)
                if temp_ != max_used_vehicle:
                    uav_2 = temp_
                    break
            # print('here2')
            req_edge_ind_max_uav = random.randint(0, len(uavToRequiredEdges[max_used_vehicle])-1)

            # insertion
            if uav_2 not in list(uavToRequiredEdges.keys()):
                uavToRequiredEdges[uav_2] = []    
            uavToRequiredEdges[uav_2].append(uavToRequiredEdges[max_used_vehicle][req_edge_ind_max_uav])
            uavToRequiredEdges[max_used_vehicle].remove(uavToRequiredEdges[max_used_vehicle][req_edge_ind_max_uav])
            u1 = max_used_vehicle
            u2 = uav_2
            # print(uavToRequiredEdges)
        # print('Here')
        newPaths = [0]*len(solution)
        newPathTimes = [0]*len(solution)
        newLastArrivalTimes = [0]*len(solution)
        newAllocation = {uav:[] for uav in range(len(solution))}

        # print('here3')
        # print(f"uav 1 : {u1}, uav 2: {u2}")
        # print(f"Before swapping or insertion mapping : {uavToRequiredEdges}")
        for uav, requiredEdges in uavToRequiredEdges.items():
            if uav == u1 or uav == u2:
                # print(f"Designing paths for uav : {uav}")
                totalUavs = 1
                uavAvailableTime = np.array([0], dtype=np.float32)
                uavPaths = [0]
                uavPathTimes = [0]
                uavLAT = np.array([0], dtype=np.float32)
                uavUtilization = np.array([0], dtype=np.float32)
                uavLocation = np.array([originalLocation[uav]], dtype=np.int16)
                # print(f'uav : {uav}, uavLocation : {uavLocation}, required edges : {requiredEdges}')
                # uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges = newHeuristics(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                #                                                                     uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
                uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges = routes_based_on_allocation(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                                                                                    uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
                # print(uavPaths)
                # print('Paths generated')
                # if uavPaths[0] == 0:
                #     # print('infeasible allocation')
                #     return solution, copy_uavToRequiredEdges, copy_uavToRequiredEdges, copy_uavLastArrivalTimes, uavLastArrivalTimes
                
                if uavPaths[0] != 0:       
                    newPaths[uav] = uavPaths[0]
                else:
                    newPaths[uav] = []
                # newPathTimes[uav] = uavPathTimes[0]
                newLastArrivalTimes[uav] = uavLAT[0] 
                # print(uav_to_requirededges)
                if uav_to_requirededges:
                    newAllocation[uav] = uav_to_requirededges[0]
                else:
                    newAllocation[uav] = []
            else:
                newPaths[uav] = copy.deepcopy(solution[uav])
                # newPathTimes[uav] = uavPathTimes[0]
                newLastArrivalTimes[uav] = uavLastArrivalTimes[uav] 
                newAllocation[uav] = uavToRequiredEdges[uav]

            # print(newAllocation)
            # print(newPaths, newAllocation)
        return newPaths, copy_uavToRequiredEdges, newAllocation, copy_uavLastArrivalTimes, np.array(newLastArrivalTimes)
    elif len(solution) == 1:
        # print('Only one UAV')

        newPaths = [0]*len(solution)
        newPathTimes = [0]*len(solution)
        newLastArrivalTimes = [0]*len(solution)
        newAllocation = {uav:[] for uav in range(len(solution))}

        uav = 0

        totalUavs = 1
        uavAvailableTime = np.array([0], dtype=np.float32)
        uavPaths = [0]
        uavPathTimes = [0]
        uavLAT = np.array([0], dtype=np.float32)
        uavUtilization = np.array([0], dtype=np.float32)
        uavLocation = np.array([originalLocation[0]], dtype=np.int16)
        req_edges = uavToRequiredEdges[0][:]
        random.shuffle(req_edges)
        # print(f"uav to required edges : {uavToRequiredEdges[0]}")
        # print(f'uav : {0}, uavLocation : {uavLocation}, required edges : {uavToRequiredEdges[0]}')
        # uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges = newHeuristics(G, uavToRequiredEdges[0], depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
        #                                                                     uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
        uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges = routes_based_on_allocation(G, req_edges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                                                                                    uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
        # print(uavPaths)
        # print('Paths generated')
        if uavPaths[0] != 0:       
            newPaths[uav] = uavPaths[0]
        else:
            newPaths[uav] = []
        # newPathTimes[uav] = uavPathTimes[0]
        newLastArrivalTimes[uav] = uavLAT[0] 
        # print(uav_to_requirededges)
        if uav_to_requirededges:
            newAllocation[uav] = uav_to_requirededges[0]
        else:
            newAllocation[uav] = []

        return newPaths, copy_uavToRequiredEdges, newAllocation, copy_uavLastArrivalTimes, np.array(newLastArrivalTimes)

    return solution, copy_uavToRequiredEdges, copy_uavToRequiredEdges, copy_uavLastArrivalTimes, uavLastArrivalTimes


# def generate_new_solution_ts(G, solution, rechargeTime, depotNodes, vehicleCapacity, vehicle_starting_depots, uavToRequiredEdges, originalLocation, uavLastArrivalTimes):
#     """
#     Function to generate new solution by performing insertion and swapping
#     operations.
#     Args:
#       solution: A list of vehicle routes. Each route is list of trip.
#                 Each trip is a list of sequence of nodes in the graph.

#     Returns:
#       A new feasible solution route.
#     """
#     copy_uavToRequiredEdges = copy.deepcopy(uavToRequiredEdges)
#     copy_uavLastArrivalTimes = np.copy(uavLastArrivalTimes)
#     # print(solution, uavToRequiredEdges)
#     # print(f"Before swapping or insertion mapping : {uavToRequiredEdges}")
#     if len(solution) >= 2:
#         u1 = -1
#         u2 = -1
#         if random.randint(1,10) >= 5 and len([edge for edge in list(uavToRequiredEdges.values()) if edge]) >= 2:
#             # Randomly swapping required edges allocation
#             # print('swapping')
#             # print(uavToRequiredEdges)
#             while True:
#                 uav_1 = random.randint(0, len(solution)-1)
#                 if uav_1 in list(uavToRequiredEdges.keys()) and len(uavToRequiredEdges[uav_1]) >= 1:
#                     break
#             while True:
#                 temp_ = random.randint(0, len(solution)-1)
#                 if temp_ != uav_1 and temp_ in list(uavToRequiredEdges.keys())  and len(uavToRequiredEdges[temp_]) >= 1:
#                     uav_2 = temp_
#                     break

#             req_edge_ind_max_uav = random.randint(0, len(uavToRequiredEdges[uav_1])-1)
#             req_edge_ind_min_uav = random.randint(0, len(uavToRequiredEdges[uav_2])-1)

#             # Swapping
#             temp = uavToRequiredEdges[uav_1][req_edge_ind_max_uav]
#             uavToRequiredEdges[uav_1][req_edge_ind_max_uav] = uavToRequiredEdges[uav_2][req_edge_ind_min_uav]
#             uavToRequiredEdges[uav_2][req_edge_ind_min_uav] = temp
#             u1 = uav_1
#             u2 = uav_2
#             # print('here1')
#             # print(uavToRequiredEdges)
#         else:
#             # print('insertion')
#             # # Random insertion of max used trip 
#             # print(uavLastArrivalTimes, np.argmax(uavLastArrivalTimes))
#             # print(uavToRequiredEdges)
#             # print(solution)
#             vehicle = []
#             while True:
#                 # max_used_vehicle = list(uavLastArrivalTimes).index(max([t for t in list(uavLastArrivalTimes) if t not in vehicle]))#np.argmax(uavLastArrivalTimes)
#                 max_used_vehicle = np.argmax(uavLastArrivalTimes)
#                 # print(max_used_vehicle, uavToRequiredEdges, uavLastArrivalTimes)
#                 if max_used_vehicle in list(uavToRequiredEdges.keys()):
#                     break
#                 else:
#                     uavLastArrivalTimes[max_used_vehicle] = 0
#                 #     vehicle.append(max_used_vehicle)
#             # print('here1')
#             while True:
#                 temp_ = random.randint(0, len(solution)-1)
#                 if temp_ != max_used_vehicle:
#                     uav_2 = temp_
#                     break
#             # print('here2')
#             req_edge_ind_max_uav = random.randint(0, len(uavToRequiredEdges[max_used_vehicle])-1)

#             # insertion
#             if uav_2 not in list(uavToRequiredEdges.keys()):
#                 uavToRequiredEdges[uav_2] = []    
#             uavToRequiredEdges[uav_2].append(uavToRequiredEdges[max_used_vehicle][req_edge_ind_max_uav])
#             uavToRequiredEdges[max_used_vehicle].remove(uavToRequiredEdges[max_used_vehicle][req_edge_ind_max_uav])
#             u1 = max_used_vehicle
#             u2 = uav_2
#             # print(uavToRequiredEdges)
#         # print('Here')
#         newPaths = [0]*len(solution)
#         newPathTimes = [0]*len(solution)
#         newLastArrivalTimes = [0]*len(solution)
#         newAllocation = {uav:[] for uav in range(len(solution))}

#         # print('here3')
#         # print(f"uav 1 : {u1}, uav 2: {u2}")
#         # print(f"Before swapping or insertion mapping : {uavToRequiredEdges}")
#         valid_allocation = True
#         for uav, requiredEdges in uavToRequiredEdges.items():
#             if uav == u1 or uav == u2:
#                 # print(f"Designing paths for uav : {uav}")
#                 totalUavs = 1
#                 uavAvailableTime = np.array([0], dtype=np.float32)
#                 uavPaths = [0]
#                 uavPathTimes = [0]
#                 uavLAT = np.array([0], dtype=np.float32)
#                 uavUtilization = np.array([0], dtype=np.float32)
#                 uavLocation = np.array([originalLocation[uav]], dtype=np.int16)
#                 # print(f'uav : {uav}, uavLocation : {uavLocation}, required edges : {requiredEdges}')
#                 # uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges = newHeuristics(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
#                 #                                                                     uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
#                 uavPaths, uavPathTimes, uavLAT, traversedEdges, numRecharges, uav_to_requirededges, valid_allocation = routes_based_on_allocation_ts(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
#                                                                                     uavAvailableTime, uavLAT, uavPaths, uavPathTimes, vehicleCapacity)
#                 # print(uavPaths)
#                 # print('Paths generated')
#                 # if uavPaths[0] == 0:
#                 #     # print('infeasible allocation')
#                 #     return solution, copy_uavToRequiredEdges, copy_uavToRequiredEdges, copy_uavLastArrivalTimes, uavLastArrivalTimes
                
#                 if not valid_allocation:
#                     return solution, copy_uavToRequiredEdges, copy_uavToRequiredEdges, copy_uavLastArrivalTimes, uavLastArrivalTimes
#                 if uavPaths[0] != 0:       
#                     newPaths[uav] = uavPaths[0]
#                 else:
#                     newPaths[uav] = []
#                 # newPathTimes[uav] = uavPathTimes[0]
#                 newLastArrivalTimes[uav] = uavLAT[0] 
#                 # print(uav_to_requirededges)
#                 if uav_to_requirededges:
#                     newAllocation[uav] = uav_to_requirededges[0]
#                 else:
#                     newAllocation[uav] = []
#             else:
#                 newPaths[uav] = copy.deepcopy(solution[uav])
#                 # newPathTimes[uav] = uavPathTimes[0]
#                 newLastArrivalTimes[uav] = uavLastArrivalTimes[uav] 
#                 newAllocation[uav] = uavToRequiredEdges[uav]

#             # print(newAllocation)
#             # print(newPaths, newAllocation)
#         return newPaths, copy_uavToRequiredEdges, newAllocation, copy_uavLastArrivalTimes, np.array(newLastArrivalTimes)
#     return solution, copy_uavToRequiredEdges, copy_uavToRequiredEdges, copy_uavLastArrivalTimes, uavLastArrivalTimes


def acceptance_probability(current_fitness, new_fitness, temperature):
    """
    Implementation of the acceptance probability function. Calculate 
    the probability of accepting a worse solution based on the current temperature.

    Args:
      current_fitness: Objective function value of the current solution routes.
      new_fitness: Objective function value of the new solution routes after swapping
                   and insertion operations.
      temperature: Metaheuristic parameter of simulated annealing method.s

    Returns:
      The acceptance probaility of new solution routes. 
    """
    delta_fitness = new_fitness - current_fitness
    if delta_fitness < 0:
        return 1.0
    return math.exp(-delta_fitness / temperature)


def simulated_annealing_for_vehicle_failures(G, initial_solution, initial_temperature, cooling_rate, max_iterations, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots, uavToRequiredEdges, uavLocation, uavLastArrivalTimes, uavarrivalTimes):
    """
    Implementation of simulated annealing method.

    Args:
      inital_solution: Initial solution route.
      initial_temperature: Metaheuristic parameter of simulated annealing method usually 100.
      cooling_rate: The factor by which the temperature decreases with each iteration.
      max_iteration: Maximum number of iterations to run simulated annealing.s

    Returns:
      A improved or same solution route.
    """
    
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    best_allocation = copy.deepcopy(uavToRequiredEdges)
    current_fitness = objective_function_vehicle_failures(G, current_solution, recharge_time, uavarrivalTimes)
    best_fitness = current_fitness
    temperature = initial_temperature
    objective_function_value = []
    swap_count = 0
    insertion_count = 0
    explored_fitness = []
    temperature_list = []
    for iteration in range(max_iterations):
        # print(f"iteration : {iteration}")
        new_solution, uavToRequiredEdges, newAllocation, uavLastArrivalTimes, newuavLastArrivalTimes = generate_new_solution(G, current_solution, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots, copy.deepcopy(uavToRequiredEdges), uavLocation, uavLastArrivalTimes)
        # print(f'after generating new solution - {new_solution}')
        new_fitness = objective_function_vehicle_failures(G, new_solution, recharge_time, uavarrivalTimes)
        # print(f"{new_fitness}")
        explored_fitness.append(new_fitness)
        
        # print('here')
        # print(uavToRequiredEdges, newAllocation)

        if (acceptance_probability(current_fitness, new_fitness, temperature) > random.random()):#random.random()):#0.9*math.exp(-temperature)):
            current_solution = copy.deepcopy(new_solution)
            current_fitness = new_fitness
            uavToRequiredEdges = copy.deepcopy(newAllocation)
            uavLastArrivalTimes = newuavLastArrivalTimes
        
        
        temperature_list.append(temperature)
        # print('here')
        # print(uavToRequiredEdges, newAllocation)

        if new_fitness < best_fitness:
            best_solution = copy.deepcopy(new_solution)
            best_fitness = new_fitness
            objective_function_value.append(new_fitness)
            best_allocation = copy.deepcopy(newAllocation)

        temperature *= cooling_rate
        # print(f"Iteration {iteration}: {best_fitness, best_solution, best_allocation, uavToRequiredEdges}")

    return best_solution, best_allocation, objective_function_value, explored_fitness, temperature_list


def simulated_annealing(G, initial_solution, initial_temperature, cooling_rate, max_iterations, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots, uavToRequiredEdges, uavLocation, uavLastArrivalTimes):
    """
    Implementation of simulated annealing method.

    Args:
      inital_solution: Initial solution route.
      initial_temperature: Metaheuristic parameter of simulated annealing method usually 100.
      cooling_rate: The factor by which the temperature decreases with each iteration.
      max_iteration: Maximum number of iterations to run simulated annealing.s

    Returns:
      A improved or same solution route.
    """
    
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    best_allocation = copy.deepcopy(uavToRequiredEdges)
    current_fitness = objective_function(G, current_solution, recharge_time)
    best_fitness = current_fitness
    temperature = initial_temperature
    objective_function_value = []
    swap_count = 0
    insertion_count = 0
    explored_fitness = []
    temperature_list = []
    for iteration in range(max_iterations):
        # print(f"iteration : {iteration}")
        new_solution, uavToRequiredEdges, newAllocation, uavLastArrivalTimes, newuavLastArrivalTimes = generate_new_solution(G, current_solution, recharge_time, depotNodes, vehicleCapacity, vehicle_starting_depots, copy.deepcopy(uavToRequiredEdges), uavLocation, uavLastArrivalTimes)
        # print(f'after generating new solution - {new_solution}')
        new_fitness = objective_function(G, new_solution, recharge_time)
        # print(f"{new_fitness}")
        explored_fitness.append(new_fitness)
        
        # print('here')
        # print(uavToRequiredEdges, newAllocation)

        if (acceptance_probability(current_fitness, new_fitness, temperature) > random.random()):#random.random()):#0.9*math.exp(-temperature)):
            current_solution = copy.deepcopy(new_solution)
            current_fitness = new_fitness
            uavToRequiredEdges = copy.deepcopy(newAllocation)
            uavLastArrivalTimes = newuavLastArrivalTimes
        
        
        temperature_list.append(temperature)
        # print('here')
        # print(uavToRequiredEdges, newAllocation)

        if new_fitness < best_fitness:
            best_solution = copy.deepcopy(new_solution)
            best_fitness = new_fitness
            objective_function_value.append(new_fitness)
            best_allocation = copy.deepcopy(newAllocation)

        temperature *= cooling_rate
        # print(f"Iteration {iteration}: {best_fitness, best_solution, best_allocation, uavToRequiredEdges}")

    return best_solution, best_allocation, objective_function_value, explored_fitness, temperature_list


def main():
    totalUavs = 2
    depotNodes = [1,5,6]
    requiredEdges = [[8, 9]]
    numNodes = 9

    G = nx.Graph()
    edges = []
    pos = {}
    reqPos = {}
    s = [1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6, 7, 8]
    t = [2, 3, 4, 6, 4, 5, 5, 7, 6, 8, 7, 8, 9]
    weights = [2.3, 2, 3, 1.5, 3.2, 2.2, 3.8, 2.6, 2.2, 1.8, 2.8, 0.8, 1]
    xData = [-2, -0.5, -1,   0, 1,  1.5, 2,   2.5, 3]
    yData = [ 0, -2,    2.5, 0, 3, -2,   0.3, 1.5, 3]
    for i in range(len(s)):
        edges.append((s[i], t[i], weights[i]))
    
    for i in range(1, numNodes+1):
        G.add_node(i)
        pos[i] =(xData[i-1], yData[i-1])
    
    node_color = ['y']*int(G.number_of_nodes())
    depot_node_color = node_color
    for i in range(1, len(node_color)+1):
        if i in depotNodes:
            depot_node_color[i-1] = 'g'
            
    G.add_weighted_edges_from(edges)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx(G,pos, node_color = node_color)
    nx.draw_networkx(G,pos, node_color = depot_node_color)
    nx.draw_networkx_edges(G, pos, edgelist=requiredEdges, width=3, alpha=0.5,
                                        edge_color="r")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()
    

    requiredEdgesCopy1 = copy.deepcopy(requiredEdges)
    vehicleCapacity = 7
    rechargeTime = 1.1
    uavLocation = np.array([1,6])
    uavLocation_copy = copy.deepcopy(uavLocation)
    uavAvailableTime = np.array([0]*totalUavs, dtype=np.float32)
    uavPaths = [0]*totalUavs
    uavPathTimes = [0]*totalUavs
    uavLastArrivalTimes = [0]*totalUavs
    uavUtilization = np.array([0]*totalUavs, dtype=np.float32)

    uavPaths, uavPathTimes, uavLastArrivalTimes, traversedEdges, numRecharges, uav_to_requirededges = newHeuristics(G, requiredEdges, depotNodes, rechargeTime, totalUavs, uavLocation, uavUtilization,
                                                                            uavAvailableTime, uavLastArrivalTimes, uavPaths, uavPathTimes, vehicleCapacity)

    # print(uavPaths, uav_to_requirededges)

    index = 0
    max_index = max([len(uavPaths[uav]) for uav in range(len(uavPaths))])
    while index <= max_index:
        for uav in range(len(uavPaths)):
            if index <= len(uavPaths[uav]) - 1:
                if requiredEdgesCopy1:
                    trip_removed = False
                    trip = uavPaths[uav][index]
                    for i in range(len(trip)-1):
                        # print(trip[i], trip[i+1])                                
                        for edge in requiredEdgesCopy1:
                            if edge == [trip[i], trip[i+1]] or edge[::-1] == [trip[i], trip[i+1]]:
                                requiredEdgesCopy1.remove(edge)
                                trip_removed = True
                    if not trip_removed:
                        if index == len(uavPaths[uav]) - 1:
                            uavPaths[uav] = uavPaths[uav][:index]            
                else:
                    uavPaths[uav] = uavPaths[uav][:index]
        index += 1

    print(uavPaths)

    # Define the initial temperature
    initial_solution = copy.deepcopy(uavPaths)
    initial_temperature = G.number_of_edges()
    vehicle_starting_depots = {}#{i : initial_solution[i][0][0] for i in range(len(initial_solution))}
    init_obj = objective_function(G, initial_solution, rechargeTime)
    # Define the cooling rate
    cooling_rate = 0.9

    # Define the maximum number of iterations
    max_iterations = 100

    # Call the simulated annealing method with the provided inputs
    best_solution, best_allocation, objective_function_value, explored_fitness, temperature_list = simulated_annealing(G, initial_solution, initial_temperature, cooling_rate, max_iterations, rechargeTime, depotNodes, vehicleCapacity, vehicle_starting_depots, uav_to_requirededges, uavLocation_copy, uavLastArrivalTimes)
    print(objective_function_value, best_solution, best_allocation)
    if objective_function_value:
        explored_fitness.insert(0, init_obj)
        temperature_list.insert(0, initial_temperature)
        # x = [i for i in range(0, len(explored_fitness))]
        # print(x)
        plt.plot(temperature_list, explored_fitness)
        plt.xlabel("Temperature")  # add X-axis label
        plt.ylabel("Maximum Trip Time")  # add Y-axis label
        plt.title("Simulated Annealing")  # add title
        plt.show()
        # Print the best solution found
        print("Initial solution: ", initial_solution)
        print("Best solution:", best_solution)
        print("Objective function value : ", objective_function_value[-1])
        print("Best Allocation : ", best_allocation)
        # print(f'Number of times swapped {swap_count} / {max_iterations}')
        # print(f'Number of times swapped {insertion_count} / {max_iterations}')

if __name__ == '__main__':
    main()

