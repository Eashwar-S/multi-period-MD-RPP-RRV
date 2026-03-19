"""
sa_mdrpp_rrv.py
===============
Simulated-Annealing metaheuristic for the Multi-Depot, Multi-Period
Route Planning Problem with Random Required Visits (MD-RPP-RRV).

Key differences from the original simulated_annealing.py:
  * Edges have traversal *probabilities* (not fixed requirement). The
    objective mirrors the Gurobi formulation:
        maximise  prob_min - beta / T_interval
    where  prob_min = min_k( sum_{e covered by k} p_e )
           beta     = makespan (max total mission time over all vehicles)
  * Novel Trip Memory Cache: sub-problem results (best trip sequence
    for a given depot + edge assignment) are memoised across SA
    iterations – avoiding repeated A* computation and making the
    algorithm scalable to large graphs.
  * Batch runner: runs over all graph pickle files and writes an Excel
    result sheet with the same columns as gurobi_results/results_all.xlsx.
"""

import math
import random
import copy
import time
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

# ─── global trip memory cache ─────────────────────────────────────────────────
# Key  : (start_depot, frozenset of canonical (min,max) edge tuples)
# Value: (trip_path_list, total_trip_time, prob_score)
#   trip_path_list  – list of node sequences (one per flight within T_interval)
#   total_trip_time – float, total traversal time (excludes recharge waits)
#   prob_score      – float, sum of p_e for unique edges covered
TRIP_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# §1  Graph & path utilities
# ══════════════════════════════════════════════════════════════════════════════

def edge_weight(G: nx.Graph, u, v) -> float:
    """Return weight of undirected edge (u,v)."""
    d = G.get_edge_data(u, v)
    if d is None:
        return float('inf')
    return d.get('weight', 1.0)


def path_time(G: nx.Graph, path: list) -> float:
    """Sum of edge weights along a node-sequence path."""
    return sum(edge_weight(G, path[i], path[i + 1])
               for i in range(len(path) - 1))


def canonical_edge(i, j):
    """Return (min, max) canonical form of an undirected edge."""
    return (min(i, j), max(i, j))


def path_covered_edges(path: list) -> set:
    """Set of canonical edges traversed by *path*."""
    return {canonical_edge(path[i], path[i + 1])
            for i in range(len(path) - 1)}


def _nearest_depot_path(G: nx.Graph, node, depot_nodes: list,
                         remaining_capacity: float):
    """
    Return (time, path) to the nearest reachable depot within remaining_capacity.
    Returns (inf, []) if none reachable.
    """
    best_t = float('inf')
    best_p = []
    for d in depot_nodes:
        if d == node:
            return 0.0, [node]
        try:
            t = nx.astar_path_length(G, node, d, weight='weight')
            if t <= remaining_capacity and t < best_t:
                best_t = t
                best_p = nx.astar_path(G, node, d, weight='weight')
        except nx.NetworkXNoPath:
            pass
    return best_t, best_p


# ══════════════════════════════════════════════════════════════════════════════
# §2  Probabilistic objective function
# ══════════════════════════════════════════════════════════════════════════════

def objective_prob(solution: list, edgeProbs: dict,
                   T_interval: float, rechargeTime: float,
                   G: nx.Graph):
    """
    Evaluate the probabilistic objective matching the Gurobi model.

    Parameters
    ----------
    solution    : list of vehicle-route lists.
                  Each vehicle route is a list of trips (node sequences).
    edgeProbs   : dict  { (min_i, max_j) : probability }
    T_interval  : float  upper bound on makespan
    rechargeTime: float  time between flights
    G           : graph (used only for edge weight lookup)

    Returns
    -------
    obj      : float  prob_min - beta/T_interval  (maximise)
    prob_min : float  min per-vehicle probability score
    beta     : float  makespan
    """
    prob_k_list = []
    mission_times = []

    for vehicle_route in solution:
        if not vehicle_route or vehicle_route == 0:
            prob_k_list.append(0.0)
            mission_times.append(0.0)
            continue

        covered = set()
        total_travel = 0.0
        num_active_flights = 0

        for trip in vehicle_route:
            if not trip:
                continue
            num_active_flights += 1
            t = path_time(G, trip)
            total_travel += t
            covered |= path_covered_edges(trip)

        prob_k = sum(edgeProbs.get(e, 0.0) for e in covered)
        prob_k_list.append(prob_k)

        beta_k = total_travel + max(0, num_active_flights - 1) * rechargeTime
        mission_times.append(beta_k)

    if not prob_k_list:
        return -float('inf'), 0.0, 0.0

    prob_min = min(prob_k_list)
    beta = max(mission_times) if mission_times else 0.0
    beta = min(beta, T_interval)
    obj = prob_min - beta / T_interval
    return obj, prob_min, beta


# ══════════════════════════════════════════════════════════════════════════════
# §3  Trip memory cache – route builder per vehicle
# ══════════════════════════════════════════════════════════════════════════════

def _build_single_vehicle_trips(G: nx.Graph, start_depot,
                                 assigned_edges: list,
                                 depot_nodes: list,
                                 vehicleCapacity: float,
                                 rechargeTime: float,
                                 numFlights: int,
                                 T_interval: float,
                                 edgeProbs: dict) -> tuple:
    """
    Greedy construction of trips for ONE vehicle starting at *start_depot*,
    required to traverse all edges in *assigned_edges*.

    Returns (trips, total_time, prob_score) where:
      trips      – list of node-sequence lists (one per flight)
      total_time – sum of all trip travel times (no recharge wait included)
      prob_score – sum of p_e for unique covered canonical edges
    """
    # Sort edges by probability descending (visit high-value edges first)
    edges_to_do = sorted(
        [list(e) for e in assigned_edges],
        key=lambda e: edgeProbs.get(canonical_edge(e[0], e[1]), 0.0),
        reverse=True
    )

    trips = []
    total_time = 0.0
    covered_canonical = set()
    current_loc = start_depot
    remaining_cap = vehicleCapacity
    current_trip = [current_loc]
    trip_time = 0.0

    while edges_to_do or (current_trip[-1] not in depot_nodes and len(current_trip) > 1):
        if edges_to_do:
            chosen_e = None
            chosen_path = None
            chosen_t = float('inf')
            chosen_idx = -1

            # Try to find a reachable edge within remaining capacity
            for idx, edge in enumerate(edges_to_do):
                for end_node in [edge[0], edge[1]]:
                    other = edge[1] if end_node == edge[0] else edge[0]
                    try:
                        t_reach = nx.astar_path_length(
                            G, current_loc, end_node, weight='weight')
                        t_traverse = edge_weight(G, end_node, other)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                    # Can we reach the edge AND return to a depot?
                    t_total = t_reach + t_traverse
                    t_ret, _ = _nearest_depot_path(
                        G, other, depot_nodes,
                        remaining_cap - t_total)
                    if t_total + t_ret <= remaining_cap and t_reach < chosen_t:
                        path_to = nx.astar_path(
                            G, current_loc, end_node, weight='weight')
                        chosen_t = t_reach
                        chosen_e = edge
                        chosen_path = path_to + [other]
                        chosen_idx = idx
                        break  # take first endpoint that works

                if chosen_e is not None:
                    break

            if chosen_e is not None:
                # Extend current trip
                current_trip += chosen_path[1:]
                trip_time += path_time(G, chosen_path)
                remaining_cap -= trip_time
                # recompute trip_time properly
                trip_time = path_time(G, current_trip)
                remaining_cap = vehicleCapacity - trip_time
                current_loc = current_trip[-1]
                covered_canonical.add(canonical_edge(chosen_e[0], chosen_e[1]))
                edges_to_do.pop(chosen_idx)
                continue

        # Cannot reach the next edge – return to depot
        t_ret, path_ret = _nearest_depot_path(
            G, current_loc, depot_nodes, remaining_cap)
        if path_ret:
            current_trip += path_ret[1:]
        if len(current_trip) > 1:
            trips.append(current_trip)
            total_time += path_time(G, current_trip)

        if not edges_to_do:
            break

        if len(trips) >= numFlights:
            break  # can't add more flights

        # Start next flight from nearest depot
        depot_loc = current_trip[-1] if current_trip else start_depot
        current_loc = depot_loc
        current_trip = [current_loc]
        trip_time = 0.0
        remaining_cap = vehicleCapacity

    # Close last open trip at a depot if needed
    if current_trip and len(current_trip) > 1 and current_trip[-1] not in depot_nodes:
        t_ret_final, path_ret_final = _nearest_depot_path(
            G, current_trip[-1], depot_nodes, vehicleCapacity)
        if path_ret_final:
            current_trip += path_ret_final[1:]
        if len(current_trip) > 1:
            trips.append(current_trip)
            total_time += path_time(G, current_trip)
    elif current_trip and len(current_trip) > 1:
        trips.append(current_trip)
        total_time += path_time(G, current_trip)

    prob_score = sum(edgeProbs.get(e, 0.0) for e in covered_canonical)
    return trips, total_time, prob_score


def cached_vehicle_route(G: nx.Graph, start_depot,
                          assigned_edges: list,
                          depot_nodes: list,
                          vehicleCapacity: float,
                          rechargeTime: float,
                          numFlights: int,
                          T_interval: float,
                          edgeProbs: dict,
                          trip_cache: dict) -> tuple:
    """
    Wrapper around _build_single_vehicle_trips that uses TRIP_CACHE.

    Cache key: (start_depot, frozenset of canonical edges)
    Returns   : (trips, total_time, prob_score)
    """
    key = (start_depot,
           frozenset(canonical_edge(e[0], e[1]) for e in assigned_edges))
    if key in trip_cache:
        return trip_cache[key]

    result = _build_single_vehicle_trips(
        G, start_depot, assigned_edges, depot_nodes,
        vehicleCapacity, rechargeTime, numFlights, T_interval, edgeProbs)

    trip_cache[key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §4  Initial solution builder
# ══════════════════════════════════════════════════════════════════════════════

def _assign_edges_to_vehicles(G: nx.Graph, all_edges: list,
                               depot_nodes: list,
                               edgeProbs: dict) -> dict:
    """
    Greedily assign each graph edge to the vehicle whose depot can reach it
    with the lowest travel time.  Returns {vehicle_idx: [edge, ...]} dict.
    """
    allocation = {k: [] for k in range(len(depot_nodes))}
    # Precompute shortest-path lengths from every depot to every node
    depot_lengths = []
    for d in depot_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, d, weight='weight')
        depot_lengths.append(lengths)

    for edge in all_edges:
        u, v = edge[0], edge[1]
        best_cost = float('inf')
        best_k = 0
        for k, lengths in enumerate(depot_lengths):
            cost = min(lengths.get(u, float('inf')),
                       lengths.get(v, float('inf')))
            # Bias towards high-probability edges for vehicles closer to them
            prob = edgeProbs.get(canonical_edge(u, v), 0.0)
            score = cost - prob  # lower is better
            if score < best_cost:
                best_cost = score
                best_k = k
        allocation[best_k].append(list(edge))

    return allocation


def build_initial_solution(G: nx.Graph, depot_nodes: list,
                            edgeProbs: dict,
                            vehicleCapacity: float,
                            rechargeTime: float,
                            numFlights: int,
                            T_interval: float,
                            trip_cache: dict) -> tuple:
    """
    Build an initial feasible solution.

    Returns
    -------
    solution         – list[list[list]] vehicle routes
    edge_allocation  – dict {vehicle_idx: [canonical edges]}
    mission_times    – list of per-vehicle total mission times
    """
    all_edges = [list(e) for e in G.edges()]
    edge_allocation = _assign_edges_to_vehicles(
        G, all_edges, depot_nodes, edgeProbs)

    solution = []
    mission_times = []
    for k, depot in enumerate(depot_nodes):
        assigned = edge_allocation[k]
        trips, total_t, _ = cached_vehicle_route(
            G, depot, assigned, depot_nodes,
            vehicleCapacity, rechargeTime, numFlights, T_interval,
            edgeProbs, trip_cache)
        solution.append(trips if trips else [[depot]])
        mission_times.append(total_t + max(0, len(trips) - 1) * rechargeTime)

    return solution, edge_allocation, mission_times


# ══════════════════════════════════════════════════════════════════════════════
# §5  Neighbourhood operators
# ══════════════════════════════════════════════════════════════════════════════

def _neighbour_swap_or_insert(edge_allocation: dict,
                               mission_times: list) -> tuple:
    """
    Generate a neighbouring edge allocation by either:
      - 50% chance: swap one random edge between two random vehicles.
      - 50% chance: move one edge from the busiest vehicle to another.

    Returns a *new* edge_allocation dict (deep copy with one change).
    """
    new_alloc = copy.deepcopy(edge_allocation)
    vehicles = list(new_alloc.keys())

    # Pick which operation
    if random.random() < 0.5 and len(vehicles) >= 2:
        # ── SWAP ─────────────────────────────────────────────────────────────
        v1, v2 = random.sample(vehicles, 2)
        if new_alloc[v1] and new_alloc[v2]:
            i1 = random.randrange(len(new_alloc[v1]))
            i2 = random.randrange(len(new_alloc[v2]))
            new_alloc[v1][i1], new_alloc[v2][i2] = (
                new_alloc[v2][i2], new_alloc[v1][i1])
            return new_alloc, (v1, v2)
    else:
        # ── INSERT (move from busiest vehicle) ───────────────────────────────
        if len(vehicles) >= 2:
            v_busy = int(np.argmax(mission_times))
            candidates = [v for v in vehicles if v != v_busy]
            v_target = random.choice(candidates)
            if new_alloc[v_busy]:
                idx = random.randrange(len(new_alloc[v_busy]))
                edge = new_alloc[v_busy].pop(idx)
                new_alloc[v_target].append(edge)
                return new_alloc, (v_busy, v_target)

    # Fallback: random edge shuffle between two vehicles
    v1, v2 = random.sample(vehicles, 2)
    if new_alloc[v1]:
        idx = random.randrange(len(new_alloc[v1]))
        edge = new_alloc[v1].pop(idx)
        new_alloc[v2].append(edge)
    return new_alloc, (v1, v2)


def rebuild_solution(G: nx.Graph, depot_nodes: list,
                     edge_allocation: dict,
                     vehicleCapacity: float,
                     rechargeTime: float,
                     numFlights: int,
                     T_interval: float,
                     edgeProbs: dict,
                     trip_cache: dict,
                     changed_vehicles: tuple) -> tuple:
    """
    Rebuild routes only for vehicles in *changed_vehicles* (cache used).
    Returns (solution, mission_times).
    """
    # We need a full solution, so we store per-vehicle results
    num_vehicles = len(depot_nodes)
    solution = [None] * num_vehicles
    mission_times = [0.0] * num_vehicles

    for k, depot in enumerate(depot_nodes):
        assigned = edge_allocation[k]
        trips, total_t, _ = cached_vehicle_route(
            G, depot, assigned, depot_nodes,
            vehicleCapacity, rechargeTime, numFlights, T_interval,
            edgeProbs, trip_cache)
        solution[k] = trips if trips else [[depot]]
        mission_times[k] = total_t + max(0, len(trips) - 1) * rechargeTime

    return solution, mission_times


# ══════════════════════════════════════════════════════════════════════════════
# §6  Acceptance probability (maximisation)
# ══════════════════════════════════════════════════════════════════════════════

def acceptance_prob_max(current_obj: float, new_obj: float,
                         temperature: float) -> float:
    """
    Metropolis acceptance for a *maximisation* problem.
    Always accept improvements (new_obj > current_obj).
    Accept worse with exp((new_obj - current_obj) / T).
    """
    delta = new_obj - current_obj
    if delta > 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(delta / temperature)


# ══════════════════════════════════════════════════════════════════════════════
# §7  Simulated annealing main loop
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing_prob(G: nx.Graph,
                              depot_nodes: list,
                              edgeProbs: dict,
                              vehicleCapacity: float,
                              rechargeTime: float,
                              numFlights: int,
                              T_interval: float,
                              initial_temperature: float = 5.0,
                              cooling_rate: float = 0.97,
                              max_iterations: int = 500,
                              trip_cache: dict = None) -> tuple:
    """
    Run Simulated Annealing for the MD-RPP-RRV probabilistic objective.

    Returns
    -------
    best_solution  : list[list[list]] of vehicle routes
    best_obj       : float  objective value (prob_min - beta/T_interval)
    best_prob      : float  prob_min at best solution
    best_beta      : float  makespan at best solution
    best_alloc     : dict   edge allocation at best solution
    """
    if trip_cache is None:
        trip_cache = {}

    # ── Build initial solution ───────────────────────────────────────────────
    solution, edge_alloc, mission_times = build_initial_solution(
        G, depot_nodes, edgeProbs, vehicleCapacity, rechargeTime,
        numFlights, T_interval, trip_cache)

    current_obj, current_prob, current_beta = objective_prob(
        solution, edgeProbs, T_interval, rechargeTime, G)

    best_solution = copy.deepcopy(solution)
    best_alloc    = copy.deepcopy(edge_alloc)
    best_obj      = current_obj
    best_prob     = current_prob
    best_beta     = current_beta

    temperature = initial_temperature

    for _ in range(max_iterations):
        # ── Generate neighbour ───────────────────────────────────────────────
        new_alloc, changed = _neighbour_swap_or_insert(edge_alloc, mission_times)
        new_solution, new_mission_times = rebuild_solution(
            G, depot_nodes, new_alloc, vehicleCapacity, rechargeTime,
            numFlights, T_interval, edgeProbs, trip_cache, changed)

        new_obj, new_prob, new_beta = objective_prob(
            new_solution, edgeProbs, T_interval, rechargeTime, G)

        # ── Accept / reject ──────────────────────────────────────────────────
        ap = acceptance_prob_max(current_obj, new_obj, temperature)
        if ap > random.random():
            solution       = new_solution
            edge_alloc     = new_alloc
            mission_times  = new_mission_times
            current_obj    = new_obj

        if new_obj > best_obj:
            best_solution = copy.deepcopy(new_solution)
            best_alloc    = copy.deepcopy(new_alloc)
            best_obj      = new_obj
            best_prob     = new_prob
            best_beta     = new_beta

        temperature *= cooling_rate

    return best_solution, best_obj, best_prob, best_beta, best_alloc


# ══════════════════════════════════════════════════════════════════════════════
# §8  Batch runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one_graph(meta: dict, graph_dir: str,
                  trip_cache: dict,
                  sa_kwargs: dict) -> dict:
    """
    Load one graph, run SA, and return a result record.

    Parameters
    ----------
    meta       : one entry from graph_params.npy
    graph_dir  : directory containing the pickle files
    trip_cache : shared memoisation dict (shared across graphs for warm starts)
    sa_kwargs  : extra keyword arguments forwarded to simulated_annealing_prob
    """
    graph_id       = int(meta['graph_id'])
    vehicleCapacity = float(meta['vehicle_capacity'])
    rechargeTime    = float(meta['recharge_time'])
    T_interval      = float(meta['time_interval']) / 2.0
    depot_nodes     = list(meta['depot_nodes'])

    pkl_path = os.path.join(graph_dir, f'{graph_id}.pickle')
    with open(pkl_path, 'rb') as fh:
        G = pickle.load(fh)

    # Edge probabilities (canonical form)
    edgeProbs = {}
    for u, v, d in G.edges(data=True):
        c = canonical_edge(u, v)
        edgeProbs[c] = d.get('prob', 0.0)

    numFlights = max(1, int(T_interval // (rechargeTime + vehicleCapacity)))
    numVehicle = len(depot_nodes)

    print(f"\n{'='*58}")
    print(f"Graph {graph_id:>3}  nodes={G.number_of_nodes():<4} "
          f"edges={G.number_of_edges():<4}  depots={depot_nodes}")
    print(f"  B={vehicleCapacity:.0f}  TR={rechargeTime:.0f}  "
          f"T={T_interval:.0f}  flights={numFlights}")

    start = time.time()
    best_sol, best_obj, best_prob, best_beta, best_alloc = simulated_annealing_prob(
        G, depot_nodes, edgeProbs,
        vehicleCapacity, rechargeTime, numFlights, T_interval,
        trip_cache=trip_cache,
        **sa_kwargs)
    elapsed = time.time() - start

    print(f"  SA done in {elapsed:.2f}s  "
          f"obj={best_obj:.6f}  prob={best_prob:.6f}  beta={best_beta:.2f}")

    return {
        'graph_id':   graph_id,
        'num_nodes':  G.number_of_nodes(),
        'num_edges':  G.number_of_edges(),
        'num_depots': numVehicle,
        'T_interval': T_interval,
        'numVehicle': numVehicle,
        'numFlights': numFlights,
        'prob':       round(best_prob, 6),
        'beta':       round(best_beta, 3),
        'obj':        round(best_obj, 6),
        'time_s':     round(elapsed, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §9  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── SA hyper-parameters ──────────────────────────────────────────────────
    SA_KWARGS = dict(
        initial_temperature=5.0,
        cooling_rate=0.97,
        max_iterations=600,
    )

    graph_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    meta_all  = np.load(os.path.join(graph_dir, 'graph_params.npy'),
                        allow_pickle=True)

    results    = []
    trip_cache: dict = {}     # shared across all graphs (warm starts)

    for idx in range(len(meta_all)):
        meta = meta_all[idx]
        try:
            if int(meta['graph_id']) == 7 or int(meta['graph_id']) == 10:
                continue
            rec = run_one_graph(meta, graph_dir, trip_cache, SA_KWARGS)
        except Exception as exc:
            graph_id = int(meta['graph_id'])
            print(f"  ERROR on graph {graph_id}: {exc}")
            rec = {
                'graph_id':   graph_id,
                'num_nodes':  None,
                'num_edges':  None,
                'num_depots': None,
                'T_interval': float(meta.get('time_interval', 0)) / 2.0,
                'numVehicle': None,
                'numFlights': None,
                'prob':       None,
                'beta':       None,
                'obj':        None,
                'time_s':     None,
            }
        results.append(rec)

    # ── Save to Excel ────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), 'sa_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'sa_results_all.xlsx')

    df = pd.DataFrame(results, columns=[
        'graph_id', 'num_nodes', 'num_edges', 'num_depots',
        'T_interval', 'numVehicle', 'numFlights',
        'prob', 'beta', 'obj', 'time_s'])

    df.to_excel(out_path, index=False)
    print(f"\n{'='*58}")
    print(f"Results saved → {out_path}")
    print(df.to_string(index=False))
