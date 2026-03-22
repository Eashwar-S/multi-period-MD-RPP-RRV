#!/usr/bin/env python3
"""
magnetic_sa_mdrpp_rrv.py
========================
Fast Magnetic-Field Metaheuristic for MD-RPP-RRV with Probabilistic Edges.

Algorithm
---------
Inner layer — ProbabilisticMagneticRouter (per vehicle / per depot):
  Adapted from demo_updated_tuning_capacity_grok_v10.py.
  Convex capacity-weighted scoring:
      score(e, u→v) = (1-w_cap)*P + w_cap*D
  where
      w_cap  = current_trip_length / vehicle_capacity   (in [0,1])
      P      = p_e * exp(-dist(u, v))                   ← probability-weighted edge magnet
               (strong pull toward high-p edges that are close to the vehicle)
      D      = exp(-dist(v, nearest_depot) / capacity)  ← depot-return urgency as battery drains
  This naturally replaces "required edges are magnets" with
  "high-probability edges are strong magnets"; uncovered edges are
  retracted to background once covered.

  Multi-trip: run trips sequentially (battery recharge between trips)
  until T_interval is exhausted or all edges with p > threshold are covered.

Outer layer — Simulated Annealing:
  State = assignment of edges to vehicles (which vehicle tries to cover which edge).
  Neighbour generation = edge-ownership swap or move-from-busiest-vehicle.
  Objective = prob_min - beta / T_interval  (matches Gurobi formulation).
  The SA is very lightweight because each inner evaluation reuses a
  precomputed all-pairs shortest-path table and runs in O(E * V) per trip.

Scalability features:
  - All-pairs Dijkstra precomputed once per graph → O(1) path-length lookups.
  - Trip-level memoisation: key=(start_depot, frozenset(edge_canonical_set)).
  - Neighbour rebuilds only the two changed vehicles (not all vehicles).
  - Fast early termination: SA stops when objective stops improving.

Batch runner: runs over all graph pickle files in graphs/ and saves results
to sa_results/magnetic_sa_results_all.xlsx with the same columns as
gurobi_results/results_all.xlsx.
"""

import math
import random
import copy
import time
import os
import pickle
from math import exp
from typing import Dict, List, Set, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# §1  Canonical-edge helpers
# ══════════════════════════════════════════════════════════════════════════════

def can(u, v) -> Tuple:
    """Canonical (min,max) form of an undirected edge."""
    return (min(u, v), max(u, v))


def path_length(G: nx.Graph, path: list) -> float:
    return sum(G[path[i]][path[i + 1]]['weight']
               for i in range(len(path) - 1))


def edges_in_path(path: list) -> Set[Tuple]:
    return {can(path[i], path[i + 1]) for i in range(len(path) - 1)}


# ══════════════════════════════════════════════════════════════════════════════
# §2  Probabilistic Magnetic Router  (single vehicle, single trip)
# ══════════════════════════════════════════════════════════════════════════════

class ProbabilisticMagneticRouter:
    """
    Builds ONE trip for a single vehicle starting at *start_depot*.

    The convex scoring adapts the capacity-aware formula from
    demo_updated_tuning_capacity_grok_v10 to use edge probabilities
    instead of hard required-edge membership.

    Parameters
    ----------
    G            : networkx Graph with 'weight' and 'prob' attributes
    dist         : precomputed all-pairs shortest-path lengths dict
    start_depot  : starting node
    depots       : list of all depot nodes
    capacity     : vehicle battery capacity
    edge_probs   : dict {canonical_edge: probability}
    target_edges : list of canonical edges this vehicle is responsible for
                   (acts as magnets; other edges can still be traversed).
    prob_threshold: edges with p < this are not targeted (default 0.0 = all)
    """

    def __init__(self, G: nx.Graph, dist: dict, start_depot: int,
                 depots: List[int], capacity: float,
                 edge_probs: Dict[Tuple, float],
                 target_edges: List[Tuple],
                 prob_threshold: float = 0.0):
        self.G = G
        self.dist = dist
        self.start_depot = start_depot
        self.depots = set(depots)
        self.capacity = float(capacity)
        self.edge_probs = edge_probs
        # Target edges sorted by probability (desc) — act as ordered magnets
        self.target_edges = sorted(
            [e for e in target_edges if edge_probs.get(e, 0.0) > prob_threshold],
            key=lambda e: edge_probs.get(e, 0.0), reverse=True)
        self._depot_list = list(depots)
        self.cached_depot_dist = {n: min((self.dist.get(n, {}).get(d, float('inf')) for d in self.depots), default=float('inf')) for n in G.nodes()}

    # ── distance helpers ──────────────────────────────────────────────────────

    def _d(self, u, v) -> float:
        return self.dist.get(u, {}).get(v, float('inf'))

    def _nearest_depot_dist(self, node: int) -> float:
        if hasattr(self, 'cached_depot_dist'):
            return self.cached_depot_dist.get(node, float('inf'))
        return min((self._d(node, d) for d in self.depots), default=float('inf'))

    def _nearest_depot(self, node: int) -> Optional[int]:
        best_d, best_l = None, float('inf')
        for d in self.depots:
            l = self._d(node, d)
            if l < best_l:
                best_l, best_d = l, d
        return best_d

    def _dist_to_frontier(self, node: int,
                           remaining_nodes: Set[int]) -> float:
        """Min distance from *node* to any endpoint of a remaining target."""
        best = float('inf')
        row = self.dist.get(node, {})
        for n in remaining_nodes:
            d = row.get(n, float('inf'))
            if d < best:
                best = d
            if best == 0.0:
                break
        return best

    # ── edge scoring ──────────────────────────────────────────────────────────

    def _score(self, v: int, edge: Tuple,
               remaining: Set[Tuple],
               remaining_nodes: Set[int],
               current_length: float,
               is_target: bool) -> float:
        """
        Convex capacity-weighted score:
            score = (1 - w_cap) * P  +  w_cap * D
        P = probability-weighted frontier proximity
        D = depot-return urgency
        """
        w_cap = min(max(current_length / max(self.capacity, 1e-9), 0.0), 1.0)

        # P: attraction toward uncovered high-prob edges
        if is_target:
            p_e = self.edge_probs.get(edge, 0.0)
            P = p_e  # direct probability reward for covering target edge
        else:
            if remaining:
                Fv = self._dist_to_frontier(v, remaining_nodes)
                # weight frontier proximity by average probability of remaining edges
                avg_p = (sum(self.edge_probs.get(e, 0.0) for e in remaining)
                         / len(remaining))
                P = avg_p * (0.0 if math.isinf(Fv) else exp(-Fv))
            else:
                P = 0.0

        # D: pull toward depot as battery drains
        d_v = self._nearest_depot_dist(v)
        D = 0.0 if math.isinf(d_v) else exp(
            -d_v / max(self.capacity, 1e-9))

        return (1.0 - w_cap) * P + w_cap * D

    # ── single trip builder ───────────────────────────────────────────────────

    def run_trip(self) -> Tuple[List[int], float, Set[Tuple]]:
        """
        Build a single trip (one battery charge) starting from start_depot.

        Returns
        -------
        route   : list of node IDs
        cost    : total travel time of this trip
        covered : set of canonical edges found in this trip that are in target_edges
        """
        route = [self.start_depot]
        current_length = 0.0
        target_set = set(self.target_edges)
        covered = set()
        depot_anchor_idx = 0

        MAX_ITER = max(len(self.G.edges()) * 6, 200)

        for _ in range(MAX_ITER):
            u = route[-1]
            remaining = target_set - covered

            if not remaining:
                break

            remaining_nodes = set()
            for (a, b) in remaining:
                remaining_nodes.add(a)
                remaining_nodes.add(b)

            candidates = []
            Fu = None  # Lazy evaluation

            for v in self.G.neighbors(u):
                w_uv = self.G[u][v]['weight']
                depot_back = self._nearest_depot_dist(v)

                # Capacity feasibility: can we step here and still return?
                if current_length + w_uv + depot_back > self.capacity + 1e-9:
                    continue

                e_sorted = can(u, v)
                is_target = e_sorted in remaining

                # Frontier-progress filter: non-target moves must bring us closer
                if not is_target:
                    if Fu is None:
                        Fu = self._dist_to_frontier(u, remaining_nodes)
                    Fv = self._dist_to_frontier(v, remaining_nodes)
                    if Fv >= Fu:
                        continue
                    # No trivial backtrack unless covering new edge
                    if len(route) >= 2 and v == route[-2]:
                        continue

                sc = self._score(v, e_sorted, remaining, remaining_nodes, current_length, is_target)
                candidates.append((not is_target, -sc, v, e_sorted, is_target, w_uv))

            if not candidates:
                # Fallback: SP to nearest endpoint of remaining edge
                best_sp = None
                best_sp_len = float('inf')
                row_u = self.dist.get(u, {})
                for (a, b) in remaining:
                    edge_w = self.dist.get(a, {}).get(b, float('inf'))
                    for t in (a, b):
                        dist_to_t = row_u.get(t, float('inf'))
                        dist_to_depot = self.cached_depot_dist.get(t, float('inf'))
                        total_needed = current_length + dist_to_t + edge_w + dist_to_depot
                        if total_needed <= self.capacity + 1e-9 and dist_to_t < best_sp_len:
                            best_sp_len = dist_to_t
                            best_sp = t
                if best_sp is not None:
                    try:
                        path_seg = nx.shortest_path(
                            self.G, u, best_sp, weight='weight')[1:]
                        for nxt in path_seg:
                            prev = route[-1]
                            w = self.G[prev][nxt]['weight']
                            if current_length + w + self._nearest_depot_dist(nxt) > self.capacity + 1e-9:
                                break
                            route.append(nxt)
                            current_length += w
                            e_s = can(prev, nxt)
                            if e_s in remaining:
                                covered.add(e_s)
                            if nxt in self.depots:
                                depot_anchor_idx = len(route) - 1
                        continue
                    except nx.NetworkXNoPath:
                        pass
                break  # Cannot make progress

            candidates.sort()
            _, _, v, e_sorted, is_target, w_uv = candidates[0]

            route.append(v)
            current_length += w_uv
            if is_target:
                covered.add(e_sorted)
            if v in self.depots:
                depot_anchor_idx = len(route) - 1

        # Close trip: return to nearest depot
        u = route[-1]
        if u not in self.depots:
            remaining_cap = self.capacity - current_length
            nd = self._nearest_depot(u)
            if nd is not None:
                dist_to_nd = self._d(u, nd)
                if dist_to_nd <= remaining_cap + 1e-9:
                    try:
                        seg = nx.shortest_path(self.G, u, nd, weight='weight')[1:]
                        # pick up any target edges along the way
                        for nxt in seg:
                            prev = route[-1]
                            w = self.G[prev][nxt]['weight']
                            route.append(nxt)
                            current_length += w
                            e_s = can(prev, nxt)
                            if e_s in (target_set - covered):
                                covered.add(e_s)
                    except nx.NetworkXNoPath:
                        pass

        assert route[0] in self.depots, "Trip must start at a depot"
        return route, current_length, covered


# ══════════════════════════════════════════════════════════════════════════════
# §3  Trip Cache
# ══════════════════════════════════════════════════════════════════════════════

class TripCache:
    """
    Cache for individual vehicle trips.
    Key: (start_depot, canonically_sorted_covered_edges, end_depot)
    Value: (route, trip_cost, covered_edges, probability_reward, usage_count, last_used_time)
    """
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.cache = {}  # start_depot -> { (edges_tuple, end_depot): (...) }
        self.time = 0
        self.size = 0
        self.hits = 0
        self.hits_accepted = 0
        self.hits_rejected = 0
        self.misses = 0

    def get_trip(self, current_depot, remaining_edges, max_cost):
        self.time += 1
        
        if current_depot not in self.cache:
            self.misses += 1
            return None, None, None, 0.0
            
        matched_keys = []
        for key, value in self.cache[current_depot].items():
            edges_tuple, end_depot = key
            route, current_length, covered, prob_reward, count, last_used = value
            
            if current_length > max_cost + 1e-9:
                continue
            
            if covered.issubset(remaining_edges):
                matched_keys.append((prob_reward, -current_length, key))
                
        if matched_keys:
            matched_keys.sort(reverse=True)
            best_key = matched_keys[0][2]
            
            route, current_length, covered, prob_reward, count, last_used = self.cache[current_depot][best_key]
            self.cache[current_depot][best_key] = (route, current_length, covered, prob_reward, count + 1, self.time)
            self.hits += 1
            return list(route), current_length, set(covered), prob_reward
            
        self.misses += 1
        return None, None, None, 0.0

    def add_trip(self, start_depot, route, current_length, covered, all_depots, prob_reward, G):
        if not covered:
            return
            
        self._add_single_trip(start_depot, route, current_length, covered, prob_reward)
        
        end_depot = route[-1]
        if start_depot != end_depot and start_depot in all_depots and end_depot in all_depots:
            rev_route = list(reversed(route))
            if all(G.has_edge(rev_route[i], rev_route[i+1]) for i in range(len(rev_route)-1)):
                rev_cost = sum(G[rev_route[i]][rev_route[i+1]]['weight'] for i in range(len(rev_route)-1))
                self._add_single_trip(end_depot, rev_route, rev_cost, covered, prob_reward)

    def _add_single_trip(self, start_depot, route, current_length, covered, prob_reward):
        end_depot = route[-1]
        edges_tuple = tuple(sorted(list(covered)))
        key = (edges_tuple, end_depot)
        
        if start_depot not in self.cache:
            self.cache[start_depot] = {}
            
        self.time += 1
        if key in self.cache[start_depot]:
            r, cl, cov, pr, count, last_used = self.cache[start_depot][key]
            self.cache[start_depot][key] = (tuple(route), current_length, set(covered), prob_reward, count + 1, self.time)
        else:
            if self.size >= self.max_size:
                lru_depot = None
                lru_key = None
                oldest_time = float('inf')
                for d, d_cache in self.cache.items():
                    for k, v in d_cache.items():
                        if v[5] < oldest_time:
                            oldest_time = v[5]
                            lru_depot = d
                            lru_key = k
                if lru_depot is not None:
                    del self.cache[lru_depot][lru_key]
                    self.size -= 1
                    
            self.cache[start_depot][key] = (tuple(route), current_length, set(covered), prob_reward, 1, self.time)
            self.size += 1

# ══════════════════════════════════════════════════════════════════════════════
# §4  Multi-trip planner for ONE vehicle (multiple battery charges)
# ══════════════════════════════════════════════════════════════════════════════

def plan_vehicle_trips(G: nx.Graph, dist: dict, depot: int,
                       depots: List[int], capacity: float,
                       recharge_time: float, T_interval: float,
                       edge_probs: Dict[Tuple, float],
                       target_edges: List[Tuple],
                       trip_cache: 'TripCache',
                       max_flights: int) -> Tuple[List[List[int]], float, Set[Tuple], float]:
    """
    Run sequential trips for one vehicle until battery budget (T_interval) exhausted
    or all target_edges are covered. Uses LRU subset mapping locally in trip_cache.
    """
    trips       = []
    trip_times  = []         # travel time per individual trip
    covered_total: Set[Tuple] = set()
    remaining    = list(target_edges)
    current_depot = depot
    time_used    = 0.0
    total_travel = 0.0
    depot_set    = set(depots)

    for flight_idx in range(max_flights):
        if not remaining:
            break
        time_budget_left = T_interval - time_used
        if flight_idx > 0:
            time_budget_left -= recharge_time  # recharge for this flight
        if time_budget_left <= 0:
            break

        remaining_set = set(remaining)

        # 1) Try exact trip from Cache
        route_c, cost_c, cov_c, prob_k_c = trip_cache.get_trip(
            current_depot, remaining_set, min(capacity, time_budget_left))

        route = None
        trip_cost = 0.0
        covered_now = set()
        accept_cache = False

        if route_c is not None:
            # Generate a fresh route to compare
            router = ProbabilisticMagneticRouter(
                G=G, dist=dist, start_depot=current_depot,
                depots=depots, capacity=min(capacity, time_budget_left),
                edge_probs=edge_probs, target_edges=remaining)

            route_f, cost_f, cov_f = router.run_trip()
            prob_k_f = sum(edge_probs.get(e, 0.0) for e in cov_f) if cov_f else 0.0

            # Only accept cache if reward is competitive
            if prob_k_c >= prob_k_f - 1e-4:  # Tolerate tiny floating point differences
                route = route_c
                trip_cost = cost_c
                covered_now = cov_c
                accept_cache = True
                trip_cache.hits_accepted += 1
            else:
                route = route_f
                trip_cost = cost_f
                covered_now = cov_f
                trip_cache.hits_rejected += 1
        else:
            # 2) Calculate fresh directly
            router = ProbabilisticMagneticRouter(
                G=G, dist=dist, start_depot=current_depot,
                depots=depots, capacity=min(capacity, time_budget_left),
                edge_probs=edge_probs, target_edges=remaining)

            route, trip_cost, covered_now = router.run_trip()

        # Save to Cache
        if trip_cost > 0 and len(route) > 1 and covered_now and not accept_cache:
            prob_k = sum(edge_probs.get(e, 0.0) for e in covered_now)
            trip_cache.add_trip(current_depot, route, trip_cost, covered_now, depot_set, prob_k, G)

        if trip_cost > 0 and len(route) > 1:
            # ── ASSERT: trip does not exceed vehicle battery capacity ──────
            actual_trip_cost = path_length(G, route)
            assert actual_trip_cost <= capacity + 1e-6, (
                f"Vehicle starting at depot={depot}, flight {flight_idx}: "
                f"trip cost {actual_trip_cost:.4f} exceeds battery capacity {capacity:.4f}. "
                f"Route: {route}")

            trips.append(route)
            trip_times.append(actual_trip_cost)
            total_travel += actual_trip_cost
            time_used    += actual_trip_cost
            if flight_idx > 0:
                time_used += recharge_time
            covered_total |= covered_now
            remaining = [e for e in remaining if e not in covered_total]
            current_depot = route[-1] if route[-1] in depot_set else depot
        else:
            # No useful trip — try a different start depot (retargeting)
            alt_depots = [d for d in depots if d != current_depot]
            improved = False
            for alt in sorted(alt_depots,
                              key=lambda d: dist.get(current_depot, {}).get(d, float('inf'))):
                dist_to_alt = dist.get(current_depot, {}).get(alt, float('inf'))
                if dist_to_alt >= capacity or dist_to_alt >= time_budget_left:
                    continue

                cap_for_alt = min(capacity - dist_to_alt, time_budget_left - dist_to_alt)
                route2_c, cost2_c, covered2_c, prob_k2_c = trip_cache.get_trip(alt, remaining_set, cap_for_alt)

                route2 = None
                cost2 = 0.0
                covered2 = set()
                accept_cache2 = False

                if route2_c is not None:
                    router2 = ProbabilisticMagneticRouter(
                        G=G, dist=dist, start_depot=alt,
                        depots=depots,
                        capacity=cap_for_alt,
                        edge_probs=edge_probs, target_edges=remaining)
                    route2_f, cost2_f, covered2_f = router2.run_trip()
                    prob_k2_f = sum(edge_probs.get(e, 0.0) for e in covered2_f) if covered2_f else 0.0

                    if prob_k2_c >= prob_k2_f - 1e-4:
                        route2 = route2_c
                        cost2 = cost2_c
                        covered2 = covered2_c
                        accept_cache2 = True
                        trip_cache.hits_accepted += 1
                    else:
                        route2 = route2_f
                        cost2 = cost2_f
                        covered2 = covered2_f
                        trip_cache.hits_rejected += 1
                else:
                    router2 = ProbabilisticMagneticRouter(
                        G=G, dist=dist, start_depot=alt,
                        depots=depots,
                        capacity=cap_for_alt,
                        edge_probs=edge_probs, target_edges=remaining)
                    route2, cost2, covered2 = router2.run_trip()

                if covered2 and len(route2) > 1 and not accept_cache2:
                    prob_k2 = sum(edge_probs.get(e, 0.0) for e in covered2)
                    trip_cache.add_trip(alt, route2, cost2, covered2, depot_set, prob_k2, G)

                if covered2 and len(route2) > 1:
                    try:
                        dead_path = nx.shortest_path(G, current_depot, alt, weight='weight')
                    except nx.NetworkXNoPath:
                        continue
                    full_route = dead_path + route2[1:]
                    full_cost  = dist_to_alt + cost2

                    # ── ASSERT: deadhead+trip within capacity ─────────────
                    assert full_cost <= capacity + 1e-6, (
                        f"Deadhead+trip cost {full_cost:.4f} exceeds capacity {capacity:.4f}")

                    trips.append(full_route)
                    trip_times.append(full_cost)
                    total_travel += full_cost
                    time_used    += full_cost
                    if flight_idx > 0:
                        time_used += recharge_time
                    covered_total |= covered2
                    remaining = [e for e in remaining if e not in covered_total]
                    current_depot = full_route[-1] if full_route[-1] in depot_set else depot
                    improved = True
                    break
            if not improved:
                break  # Cannot make progress from any depot

    # ── ASSERT: trip-to-trip continuity ──────────────────────────────────────
    for i in range(len(trips) - 1):
        assert trips[i][-1] == trips[i + 1][0], (
            f"Continuity violation between trip {i} and trip {i+1}: "
            f"trip {i} ends at node {trips[i][-1]} but trip {i+1} starts "
            f"at node {trips[i+1][0]}.")

    n_trips      = len(trips)
    mission_time = total_travel + max(0, n_trips - 1) * recharge_time

    return trips, total_travel, covered_total, mission_time, trip_times


# ══════════════════════════════════════════════════════════════════════════════
# §5  Objective function (matches Gurobi formulation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_objective(all_trips: List[List[List[int]]],
                      edge_probs: Dict[Tuple, float],
                      mission_times: List[float],
                      T_interval: float
                      ) -> Tuple[float, float, float, float, Set[Tuple]]:
    """
    Compute:  obj = prob_min - beta / T_interval

    Per-vehicle (matches Gurobi):
      prob_k   = sum of p_e for UNIQUE canonical edges covered by vehicle k.
      prob_min = min_k(prob_k)   — the objective component.

    Global coverage (single-count verification):
      global_covered = UNION of all edges covered by any vehicle.
      global_prob    = sum of p_e for global_covered.
      An edge covered by multiple vehicles still contributes p_e only ONCE
      to global_prob (union, not sum).

    Returns (obj, prob_min, beta, global_prob, global_covered)
    """
    prob_k_list   = []
    global_covered: Set[Tuple] = set()   # union across ALL vehicles

    for trips in all_trips:
        # per-vehicle unique edges
        covered_k: Set[Tuple] = set()
        for trip in trips:
            covered_k |= edges_in_path(trip)
        prob_k = sum(edge_probs.get(e, 0.0) for e in covered_k)
        prob_k_list.append(prob_k)
        global_covered |= covered_k      # accumulate union

    if not prob_k_list:
        return -float('inf'), 0.0, 0.0, 0.0, set()

    # prob_min: per-vehicle minimum (objective, matches Gurobi)
    prob_min = min(prob_k_list)

    # global_prob: each edge counted ONCE even if multiple vehicles covered it
    global_prob = sum(edge_probs.get(e, 0.0) for e in global_covered)

    beta = max(mission_times) if mission_times else 0.0
    beta = min(beta, T_interval)
    # obj  = prob_min - beta / T_interval
    obj  = global_prob #- beta / T_interval
    return obj, prob_min, beta, global_prob, global_covered


# ══════════════════════════════════════════════════════════════════════════════
# §6  Initial edge allocation (greedy probability-weighted assignment)
# ══════════════════════════════════════════════════════════════════════════════

def greedy_allocate(G: nx.Graph, dist: dict, depots: List[int],
                    edge_probs: Dict[Tuple, float]) -> Dict[int, List[Tuple]]:
    """
    Assign each edge to the vehicle with the best (reach × probability) score.
    Balances coverage by rotating which vehicle gets the highest-prob unassigned edge.
    """
    allocation: Dict[int, List[Tuple]] = {k: [] for k in range(len(depots))}
    # Sort edges by probability descending → higher-value edges claimed first
    all_edges = sorted(edge_probs.keys(), key=lambda e: edge_probs[e], reverse=True)

    # Precompute depot-to-node distances
    depot_dists = [dist.get(d, {}) for d in depots]

    for e in all_edges:
        a, b = e
        best_k, best_score = 0, float('inf')
        for k, row in enumerate(depot_dists):
            reach = min(row.get(a, float('inf')), row.get(b, float('inf')))
            p = edge_probs.get(e, 0.0)
            # Lower is better: close AND high prob edges preferred
            score = reach / max(p, 1e-6)
            if score < best_score:
                best_score = score
                best_k = k
        allocation[best_k].append(e)

    return allocation


# ══════════════════════════════════════════════════════════════════════════════
# §7  SA neighbourhood operator
# ══════════════════════════════════════════════════════════════════════════════

def neighbour_allocation(allocation: Dict[int, List[Tuple]],
                          mission_times: List[float],
                          covered_by_vehicle: Dict[int, Set[Tuple]],
                          uncovered_global: List[Tuple],
                          edge_probs: Dict[Tuple, float]) -> Tuple[Dict, Tuple[int, int]]:
    """
    Generate a neighbouring allocation using coverage-aware perturbations.
    Returns (new_alloc, (v1, v2)) — only v1 and v2 need rebuilding.
    """
    new_alloc = copy.deepcopy(allocation)
    vehicles = list(new_alloc.keys())
    n = len(vehicles)
    if n < 2:
        return new_alloc, (0, 0)
    
    r = random.random()
    
    # ── 1. Push Uncovered (40% probability) ──
    # Actively force an unreached edge to a different vehicle
    if r < 0.4 and uncovered_global:
        # Choose the uncovered edge with highest probability
        e_new = max(uncovered_global, key=lambda x: edge_probs.get(x, 0.0))
        
        v_owner = None
        for v in vehicles:
            if e_new in new_alloc[v]:
                v_owner = v
                break
                
        if v_owner is not None:
            others = [v for v in vehicles if v != v_owner]
            v_target = random.choice(others)
            
            if new_alloc[v_target]:
                e_old = min(new_alloc[v_target], key=lambda x: edge_probs.get(x, 0.0))
                if edge_probs.get(e_new, 0.0) > edge_probs.get(e_old, 0.0):
                    # Swap the highest prob uncovered edge with the lowest prob edge from the target vehicle
                    new_alloc[v_owner].remove(e_new)
                    new_alloc[v_target].remove(e_old)
                    new_alloc[v_target].append(e_new)
                    new_alloc[v_owner].append(e_old)
                    return new_alloc, (v_owner, v_target)
            
            # Fallback if no lower prob edge exists or target is empty - just insert
            new_alloc[v_owner].remove(e_new)
            new_alloc[v_target].append(e_new)
            return new_alloc, (v_owner, v_target)

    # ── 2. Rebalance Covered (40% probability) ──
    # Take an edge successfully covered by the busiest vehicle and shift it
    elif r < 0.8:
        v_busy = int(np.argmax(mission_times))
        covered_busy = list(covered_by_vehicle.get(v_busy, set()))
        if covered_busy:
            e = random.choice(covered_busy)
            if e in new_alloc[v_busy]:
                others = [v for v in vehicles if v != v_busy]
                v_target = random.choice(others)
                new_alloc[v_busy].remove(e)
                new_alloc[v_target].append(e)
                return new_alloc, (v_busy, v_target)

    # ── 3. Swap / Fallback Move (20% probability or previous steps failed) ──
    v1, v2 = random.sample(vehicles, 2)
    if new_alloc[v1] and new_alloc[v2]:
        i1 = random.randrange(len(new_alloc[v1]))
        i2 = random.randrange(len(new_alloc[v2]))
        new_alloc[v1][i1], new_alloc[v2][i2] = new_alloc[v2][i2], new_alloc[v1][i1]
        return new_alloc, (v1, v2)
    elif new_alloc[v1]:
        e = new_alloc[v1].pop(random.randrange(len(new_alloc[v1])))
        new_alloc[v2].append(e)
        return new_alloc, (v1, v2)
        
    return new_alloc, (v1, v2)


# ══════════════════════════════════════════════════════════════════════════════
# §8  Full solution builder (all vehicles)
# ══════════════════════════════════════════════════════════════════════════════

def build_full_solution(G: nx.Graph, dist: dict, depots: List[int],
                         allocation: Dict[int, List[Tuple]],
                         capacity: float, recharge_time: float,
                         T_interval: float, max_flights: int,
                         edge_probs: Dict[Tuple, float],
                         trip_cache: 'TripCache',
                         only_vehicles: Optional[Tuple[int, int]] = None
                         ) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Build routes for all vehicles (or only the two in *only_vehicles*).
    Returns (all_trips_per_vehicle, mission_times_per_vehicle).
    """
    n = len(depots)
    all_trips = [None] * n
    mission_times = [0.0] * n

    for k, depot in enumerate(depots):
        if only_vehicles is not None and k not in only_vehicles:
            # Keep cached result for unchanged vehicles (must be pre-filled by caller)
            continue
        target_edges = allocation[k]
        trips, _, _, mt, _ = plan_vehicle_trips(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, trip_cache, max_flights)
        all_trips[k] = trips
        mission_times[k] = mt

    return all_trips, mission_times


def _extract_coverage_state(all_trips: List[List[List[int]]], allocation: Dict[int, List[Tuple]]) -> Tuple[Dict[int, Set[Tuple]], List[Tuple]]:
    """Helper to determine strictly reachable vs untouchable edges across fleets."""
    covered_by_vehicle: Dict[int, Set[Tuple]] = {}
    global_covered = set()
    
    for vk, trips in enumerate(all_trips):
        cov_k = set()
        if trips:  # Handles generic cases
            for trip in trips:
                cov_k |= edges_in_path(trip)
        covered_by_vehicle[vk] = cov_k
        global_covered |= cov_k
        
    uncovered_global = []
    for vk, edges_assigned in allocation.items():
        for e in edges_assigned:
            if e not in global_covered:
                uncovered_global.append(e)
                
    return covered_by_vehicle, uncovered_global


# ══════════════════════════════════════════════════════════════════════════════
# §9  Main SA loop
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing_magnetic(G: nx.Graph,
                                   dist: dict,
                                   depots: List[int],
                                   edge_probs: Dict[Tuple, float],
                                   capacity: float,
                                   recharge_time: float,
                                   T_interval: float,
                                   max_flights: int,
                                   initial_temperature: float = 2.0,
                                   cooling_rate: float = 0.95,
                                   max_iterations: int = 300,
                                   no_improve_limit: int = 80,
                                   trip_cache: 'TripCache' = None
                                   ) -> Tuple[List, float, float, float]:
    """
    Run Magnetic-Field SA for the MD-RPP-RRV probabilistic objective.

    Returns (best_all_trips, best_obj, best_prob, best_beta)
    """
    if trip_cache is None:
        trip_cache = TripCache(max_size=100)

    # ── Initial solution ─────────────────────────────────────────────────────
    allocation = greedy_allocate(G, dist, depots, edge_probs)
    all_trips, mission_times = build_full_solution(
        G, dist, depots, allocation, capacity, recharge_time, T_interval,
        max_flights, edge_probs, trip_cache)

    current_obj, current_prob, current_beta, _, _ = compute_objective(
        all_trips, edge_probs, mission_times, T_interval)

    all_objectives = [[current_obj, 0, initial_temperature, max(mission_times)]]
    best_obj   = current_obj
    best_prob  = current_prob
    best_beta  = current_beta
    best_alloc = copy.deepcopy(allocation)
    best_trips = copy.deepcopy(all_trips)
    best_mt    = list(mission_times)

    covered_by_vehicle, uncovered_global = _extract_coverage_state(all_trips, allocation)

    temperature = initial_temperature
    no_improve  = 0

    for iteration in range(max_iterations):
        # ── Generate neighbour ──────────────────────────────────────────────
        new_alloc, (v1, v2) = neighbour_allocation(allocation, mission_times, covered_by_vehicle, uncovered_global, edge_probs)

        # Rebuild only the two changed vehicles; keep the rest
        new_all_trips  = copy.deepcopy(all_trips)
        new_mt         = list(mission_times)

        for vk in (v1, v2):
            depot = depots[vk]
            target_edges = new_alloc[vk]
            trips, _, _, mt, _ = plan_vehicle_trips(
                G, dist, depot, depots, capacity, recharge_time, T_interval,
                edge_probs, target_edges, trip_cache, max_flights)
            new_all_trips[vk] = trips
            new_mt[vk] = mt

        new_obj, new_prob, new_beta, _, _ = compute_objective(
            new_all_trips, edge_probs, new_mt, T_interval)

        new_cov_by_veh, new_uncovered = _extract_coverage_state(new_all_trips, new_alloc)

        # ── Metropolis acceptance (maximisation) ────────────────────────────
        delta = new_obj - current_obj
        ap = 1.0 if delta > 0 else (
            math.exp(delta / temperature) if temperature > 1e-10 else 0.0)

        if ap > random.random():
            allocation    = new_alloc
            all_trips     = new_all_trips
            mission_times = new_mt
            current_obj   = new_obj
            covered_by_vehicle = new_cov_by_veh
            uncovered_global   = new_uncovered

        if new_obj > best_obj + 1e-9:
            all_objectives.append([new_obj, iteration+1, temperature, max(new_mt)])
            best_obj   = new_obj
            best_prob  = new_prob
            best_beta  = new_beta
            best_alloc = copy.deepcopy(new_alloc)
            best_trips = copy.deepcopy(new_all_trips)
            best_mt    = list(new_mt)
            no_improve = 0
        else:
            no_improve += 1

        temperature *= cooling_rate

        if no_improve >= no_improve_limit:
            break  # Early termination

    return best_trips, best_obj, best_prob, best_beta#, all_objectives


# ══════════════════════════════════════════════════════════════════════════════
# §10  Detailed solution printer
# ══════════════════════════════════════════════════════════════════════════════

def print_solution_details(G: nx.Graph, best_trips: List[List[List[int]]],
                            depot_nodes: List[int],
                            edge_probs: Dict[Tuple, float],
                            capacity: float,
                            recharge_time: float,
                            T_interval: float,
                            graph_id: int) -> None:
    """
    Verbose terminal printout of all vehicle trips:
      - Route node sequence
      - Trip travel time vs vehicle capacity
      - Continuity check between consecutive trips
      - Per-vehicle probability score
      - Global covered edges (union, each counted once)
    """
    depot_set = set(depot_nodes)

    # Compute mission times properly
    all_mission_times = []
    for trips in best_trips:
        tt = sum(path_length(G, t) for t in trips)
        mt = tt + max(0, len(trips) - 1) * recharge_time
        all_mission_times.append(mt)

    _, best_prob_final, best_beta_final, global_prob_final, global_covered_final = compute_objective(
        best_trips, edge_probs, all_mission_times, T_interval)

    all_edges_canonical = {can(u, v) for u, v in G.edges()}
    n_total  = len(all_edges_canonical)
    n_covered = len(global_covered_final)
    n_uncovered = n_total - n_covered
    pct_uncovered = 100.0 * n_uncovered / max(n_total, 1)

    print(f"\n  ── Solution Details  (graph {graph_id}) ──")
    print(f"  prob_min (objective, per-vehicle min) = {best_prob_final:.6f}")
    print(f"  global_prob (union coverage, 1x per edge) = {global_prob_final:.6f}")
    print(f"  makespan beta = {best_beta_final:.2f}  T_interval = {T_interval:.0f}")
    print(f"  edges covered (globally) : {n_covered}/{n_total}  "
          f"({100-pct_uncovered:.1f}% covered, {pct_uncovered:.1f}% uncovered)")

    for k, trips in enumerate(best_trips):
        depot_k     = depot_nodes[k]
        n_flights   = len(trips)
        covered_k: Set[Tuple] = set()
        for t in trips:
            covered_k |= edges_in_path(t)
        prob_k = sum(edge_probs.get(e, 0.0) for e in covered_k)
        total_t = sum(path_length(G, t) for t in trips)
        mission_t = total_t + max(0, n_flights - 1) * recharge_time

        print(f"\n  Vehicle {k}  (home depot={depot_k})  "
              f"flights={n_flights}  prob_k={prob_k:.4f}  "
              f"mission_time={mission_t:.2f}")

        for f_idx, trip in enumerate(trips):
            t_cost = path_length(G, trip)
            ok_cap  = "OK" if t_cost <= capacity + 1e-6 else "OVER CAPACITY!"
            # Continuity: does this trip start where the previous ended?
            if f_idx > 0:
                prev_end   = trips[f_idx - 1][-1]
                this_start = trip[0]
                cont_ok = "OK" if prev_end == this_start else f"BROKEN (prev ended {prev_end})"
            else:
                cont_ok = "(first trip)"
            print(f"    Flight {f_idx}: {trip}")
            print(f"      time={t_cost:.3f} / capacity={capacity:.0f}  [{ok_cap}]  "
                  f"continuity={cont_ok}")

    print(f"  {'─'*54}")


# ══════════════════════════════════════════════════════════════════════════════
# §11  Per-graph runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one_graph(meta: dict, graph_dir: str,
                  global_cache: dict,
                  sa_kwargs: dict) -> dict:
    """Load one graph, run the magnetic SA, return a result record."""
    graph_id       = int(meta['graph_id'])
    capacity       = float(meta['vehicle_capacity'])
    recharge_time  = float(meta['recharge_time'])
    T_interval     = float(meta['time_interval']) / 2.0
    depot_nodes    = list(meta['depot_nodes'])

    pkl_path = os.path.join(graph_dir, f'{graph_id}.pickle')
    with open(pkl_path, 'rb') as fh:
        G = pickle.load(fh)

    # Build edge probability dict (canonical form)
    edge_probs: Dict[Tuple, float] = {}
    for u, v, d in G.edges(data=True):
        edge_probs[can(u, v)] = d.get('prob', 0.0)

    # Precompute all-pairs shortest-path lengths once per graph
    dist: Dict[int, Dict[int, float]] = dict(
        nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    max_flights = max(1, int(T_interval // max(recharge_time + capacity, 1e-9)))
    numVehicle  = len(depot_nodes)

    print(f"\n{'='*58}")
    print(f"Graph {graph_id:>3}  nodes={G.number_of_nodes():<4} "
          f"edges={G.number_of_edges():<4}  depots={depot_nodes}")
    print(f"  B={capacity:.0f}  TR={recharge_time:.0f}  "
          f"T={T_interval:.0f}  max_flights={max_flights}")

    t0 = time.time()

    # Per-graph trip cache (sub-dict inside global_cache)
    if graph_id not in global_cache:
        global_cache[graph_id] = TripCache(max_size=50000)
    trip_cache = global_cache[graph_id]

    best_trips, best_obj, best_prob, best_beta = simulated_annealing_magnetic(
        G, dist, depot_nodes, edge_probs,
        capacity, recharge_time, T_interval, max_flights,
        trip_cache=trip_cache,
        **sa_kwargs)

    elapsed = time.time() - t0
    
    total_queries = trip_cache.hits + trip_cache.misses
    hit_rate = (trip_cache.hits / total_queries * 100) if total_queries > 0 else 0.0
    accepted_rate = (trip_cache.hits_accepted / trip_cache.hits * 100) if trip_cache.hits > 0 else 0.0
    
    print(f"  Done in {elapsed:.2f}s  "
          f"obj={best_obj:.6f}  prob={best_prob:.6f}  beta={best_beta:.2f}")
    print(f"  Cache hits: {trip_cache.hits} | Cache misses: {trip_cache.misses} | Hit rate: {hit_rate:.2f}%")
    print(f"  Cache accepts: {trip_cache.hits_accepted} | Cache rejects: {trip_cache.hits_rejected} | Accept rate: {accepted_rate:.2f}%")

    # ── Detailed solution printout ─────────────────────────────────────────
    print_solution_details(G, best_trips, depot_nodes, edge_probs,
                           capacity, recharge_time, T_interval, graph_id)

    # ── Compute global coverage metrics ───────────────────────────────────
    all_mission_times = [
        sum(path_length(G, t) for t in trips) + max(0, len(trips) - 1) * recharge_time
        for trips in best_trips]
    _, _, _, global_prob, global_covered = compute_objective(
        best_trips, edge_probs, all_mission_times, T_interval)

    all_edges_canonical = {can(u, v) for u, v in G.edges()}
    n_total    = len(all_edges_canonical)
    n_covered  = len(global_covered)
    pct_uncovered = round(100.0 * (n_total - n_covered) / max(n_total, 1), 2)


    # plt.plot([all_obj[i][1] for i in range(len(all_obj))], [all_obj[i][0] for i in range(len(all_obj))])

    # plt.show()

    # plt.plot([all_obj[i][1] for i in range(len(all_obj))], [all_obj[i][2] for i in range(len(all_obj))])

    # plt.show()

    # plt.plot([all_obj[i][1] for i in range(len(all_obj))], [all_obj[i][3] for i in range(len(all_obj))])

    # plt.show()

    return {
        'graph_id':            graph_id,
        'num_nodes':           G.number_of_nodes(),
        'num_edges':           G.number_of_edges(),
        'num_depots':          numVehicle,
        'T_interval':          T_interval,
        'numVehicle':          numVehicle,
        'numFlights':          max_flights,
        'prob':                round(best_prob, 6),
        'beta':                round(best_beta, 3),
        'obj':                 round(best_obj, 6),
        'time_s':              round(elapsed, 3),
        'pct_uncovered_edges': pct_uncovered,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §12  Batch runner & entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── SA hyper-parameters ──────────────────────────────────────────────────
    SA_KWARGS = dict(
        initial_temperature = 200.0,
        cooling_rate        = 0.9,
        max_iterations      = 500,
        no_improve_limit    = 200,
    )

    graph_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    meta_all  = np.load(os.path.join(graph_dir, 'graph_params.npy'),
                        allow_pickle=True)
    # print(meta_all)
    results: List[dict] = []
    global_cache: dict  = {}   # trip_cache shared across graphs for warm starts

    for idx in range(len(meta_all)):
        if idx == len(meta_all)-1:
            meta = meta_all[idx]
            try:
                rec = run_one_graph(meta, graph_dir, global_cache, SA_KWARGS)
            except Exception as exc:
                gid = int(meta['graph_id'])
                print(f"  ERROR on graph {gid}: {exc}")
                import traceback; traceback.print_exc()
                rec = {
                    'graph_id':            gid,
                    'num_nodes':           None,
                    'num_edges':           None,
                    'num_depots':          None,
                    'T_interval':          float(meta.get('time_interval', 0)) / 2.0,
                    'numVehicle':          None,
                    'numFlights':          None,
                    'prob':                None,
                    'beta':                None,
                    'obj':                 None,
                    'time_s':              None,
                    'pct_uncovered_edges': None,
                }
            results.append(rec)

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(__file__), 'sa_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'magnetic_sa_results_all_3.xlsx')

    df = pd.DataFrame(results, columns=[
        'graph_id', 'num_nodes', 'num_edges', 'num_depots',
        'T_interval', 'numVehicle', 'numFlights',
        'prob', 'beta', 'obj', 'time_s', 'pct_uncovered_edges'])

    df.to_excel(out_path, index=False)
    print(f"\n{'='*58}")
    print(f"Results saved → {out_path}")
    print(df.to_string(index=False))
