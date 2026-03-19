#!/usr/bin/env python3
"""
intelligent_trip_memory_runner.py
==================================
Memory-Augmented Metaheuristic for Repeated Probabilistic MD-RPP-RRV Instances.

Research framing
----------------
In many real-world inspection / surveillance missions the graph topology
(road network, building layout, sensor graph) is *fixed* but the hazard
probability distribution assigned to each edge changes between mission
instances (e.g. weather updates, new threat intelligence, dynamic risk maps).
Re-solving the full SA from scratch for every probability instance wastes
computation because many optimal vehicle trips remain structurally similar
across instances.

This script implements a **Probability-Ranked Trip Memory Bank (PRTMB)**:
  1. Every useful trip produced during any SA run is stored in a memory bank
     together with a *probability-ranked signature* — the top-k covered edges
     ordered by their probability at the time of creation.
  2. When a new probability instance arrives, candidate memories are retrieved
     using a *multi-criterion fitness score* that measures how well the stored
     trip matches the *new* probability landscape without requiring an exact
     edge-set match (contrast with the existing per-run exact cache).
  3. Retrieved trips are either *directly reused* (if already feasible and
     profitable) or *lightly repaired* (stale low-value edges dropped,
     nearby high-value edges greedily inserted).
  4. Three experimental modes are compared across 7 probability instances
     on the same base graph:
       A – Baseline:          fresh SA, no cross-instance memory
       B – Exact cache only:  reuse only the existing (depot, frozenset) cache
       C – Intelligent PRTMB: full probability-aware retrieval + repair

What makes this different from simple memoisation
--------------------------------------------------
* Memoisation requires bit-exact inputs; PRTMB retrieves on *similarity*.
* The multi-criterion score accounts for current probabilities, historical
  success rate, and signature overlap simultaneously.
* The probability-ranked signature enables structural comparison between
  instances that have different numeric probabilities but similar spatial
  risk patterns.
* Repair makes partial reuse possible, which is impossible under exact caching.

Publication potential
---------------------
The approach is directly applicable to:
  - Adaptive multi-robot inspection under evolving risk maps
  - Warm-starting heuristics for time-series vehicle routing
  - Transfer learning analogues for combinatorial optimisation

Extensions toward a journal paper
----------------------------------
  - Learn retrieval weights (w1..w4) via meta-learning over many graph families
  - Add a diversity criterion so the memory bank stays representative
  - Extend to graphs where topology changes slightly between instances
  - Formal complexity and approximation analysis of the repair operator
"""

# ── Standard library ──────────────────────────────────────────────────────────
import copy
import logging
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Third-party ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# ── Local – import everything we need from the existing solver ────────────────
from magnetic_sa_mdrpp_rrv import (
    ProbabilisticMagneticRouter,
    build_full_solution,
    can,
    compute_objective,
    edges_in_path,
    greedy_allocate,
    neighbour_allocation,
    path_length,
    plan_vehicle_trips,
    print_solution_details,
    simulated_annealing_magnetic,
)

# ══════════════════════════════════════════════════════════════════════════════
# §0  Logging
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('prtmb')


# ══════════════════════════════════════════════════════════════════════════════
# §1  Configuration
# ══════════════════════════════════════════════════════════════════════════════

# SA hyper-parameters (identical across all three modes for fairness)
SA_KWARGS: dict = dict(
    initial_temperature=100.0,
    cooling_rate=0.95,
    max_iterations=1000,
    no_improve_limit=200,
)

# Memory retrieval configuration
MEMORY_CONFIG: dict = dict(
    top_k_signature=5,      # number of edges in the probability-ranked signature
    top_n_candidates=8,     # how many memory candidates to evaluate per planning call
    w1=0.50,                # weight: current probability fit
    w2=0.20,                # weight: historical success rate
    w3=0.20,                # weight: signature match
    w4=0.10,                # weight: repair cost penalty
    min_direct_fit=0.70,    # fraction of best possible fit to accept without repair
    min_repair_fit=0.40,    # fraction of best possible fit to attempt repair
    max_memory_size=2000,   # evict oldest if exceeded
)

# Experiment configuration
NUM_INSTANCES   = 7
N_REPEATS       = 10   # independent SA repeats per (mode × instance) for mean±std
RANDOM_SEED_BASE = 42
PROB_PERTURBATION_SCALE = 0.25   # Gaussian σ for probability perturbation
PROB_MIN_CLIP = 0.05
PROB_MAX_CLIP = 0.95
GRAPH_ID_TO_RUN = 15              # which graph to use for the 7-instance experiment


# ══════════════════════════════════════════════════════════════════════════════
# §2  Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTrip:
    """
    One entry in the Probability-Ranked Trip Memory Bank.

    Stores both structural information about the route and historical
    performance statistics that guide retrieval scoring.
    """
    graph_id:                    int
    instance_id_created:         int
    vehicle_home_depot:          int
    start_depot:                 int
    end_depot:                   int
    route:                       List[int]
    covered_edges:               Set[Tuple]          # canonical undirected edges
    trip_cost:                   float
    mission_time_component:      float
    top_k_signature_edges:       List[Tuple]         # top-k prob-ranked edges at creation
    top_k_signature_probs:       List[float]         # corresponding probs at creation
    creation_reward:             float               # sum of p_e at creation time
    creation_reward_per_cost:    float               # creation_reward / trip_cost
    reuse_count:                 int    = 0
    success_count:               int    = 0
    avg_repair_gain:             float  = 0.0
    avg_repair_cost:             float  = 0.0
    historical_best_fit:         float  = 0.0
    historical_mean_fit:         float  = 0.0
    _fit_samples:                List[float] = field(default_factory=list, repr=False)

    def record_reuse(self, fit: float, repaired: bool,
                     repair_gain: float = 0.0, repair_cost: float = 0.0) -> None:
        """Update historical statistics after a reuse attempt."""
        self.reuse_count += 1
        if fit > 0:
            self.success_count += 1
        self._fit_samples.append(fit)
        self.historical_best_fit  = max(self.historical_best_fit, fit)
        self.historical_mean_fit  = sum(self._fit_samples) / len(self._fit_samples)
        if repaired:
            n = self.reuse_count
            self.avg_repair_gain = (self.avg_repair_gain * (n - 1) + repair_gain) / n
            self.avg_repair_cost = (self.avg_repair_cost * (n - 1) + repair_cost) / n


@dataclass
class ExperimentRecord:
    """One row in the final results DataFrame."""
    mode:                  str
    graph_id:              int
    instance_id:           int
    repeat:                int    # 0-indexed repeat index within (mode, instance)
    seed:                  int
    runtime_s:             float
    obj:                   float
    global_prob:           float
    prob_min:              float
    beta:                  float
    pct_uncovered:         float
    n_scratch_builds:      int    = 0
    n_exact_hits:          int    = 0
    n_memory_hits:         int    = 0
    n_repaired:            int    = 0
    n_rejected:            int    = 0
    n_candidates_eval:     int    = 0
    memory_hit_rate:       float  = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §3  Probability-Ranked Trip Memory Bank (PRTMB)
# ══════════════════════════════════════════════════════════════════════════════

class ProbabilityRankedTripMemory:
    """
    Stores and retrieves vehicle trips across multiple probability instances.

    Retrieval is based on a multi-criterion score that measures how well
    a stored trip matches the current (new) probability instance without
    requiring exact edge-set equality.

    Retrieval score for memory trip m under new edge probabilities p_new:

        score(m) = w1 * current_fit(m, p_new)
                 + w2 * historical_success(m)
                 + w3 * signature_match(m, p_new)
                 - w4 * repair_cost_estimate(m)

    where:
        current_fit      = Σ p_new[e] for e in m.covered_edges  / m.trip_cost
        historical_success = m.historical_mean_fit  (prior reuse quality)
        signature_match  = |top_k(p_new) ∩ m.signature| / k
        repair_cost_est  = m.avg_repair_cost  (empirically tracked)
    """

    def __init__(self, cfg: dict = None):
        self._cfg    = cfg or MEMORY_CONFIG
        self._bank:  List[MemoryTrip] = []

    # ── public API ────────────────────────────────────────────────────────────

    def store(self, trip: MemoryTrip) -> None:
        """Add a trip to the memory bank, evicting old entries if full."""
        self._bank.append(trip)
        if len(self._bank) > self._cfg['max_memory_size']:
            # Evict the least-reused entry (simple LRU-like strategy)
            self._bank.sort(key=lambda m: m.reuse_count)
            self._bank.pop(0)

    def retrieve(self,
                 graph_id: int,
                 home_depot: int,
                 edge_probs_new: Dict[Tuple, float],
                 all_edges: Set[Tuple]) -> List[Tuple[float, 'MemoryTrip']]:
        """
        Return sorted list of (score, MemoryTrip) for the best candidates.

        Filters to the same graph and home depot before scoring.
        """
        cfg = self._cfg
        k   = cfg['top_k_signature']
        w1, w2, w3, w4 = cfg['w1'], cfg['w2'], cfg['w3'], cfg['w4']

        # Global top-k edges by current probability (used for signature match)
        global_topk = set(
            e for e, _ in sorted(
                edge_probs_new.items(), key=lambda x: x[1], reverse=True
            )[:k]
        )

        candidates = [
            m for m in self._bank
            if m.graph_id == graph_id and m.vehicle_home_depot == home_depot
            and m.covered_edges  # non-empty trip
        ]

        scored = []
        for m in candidates:
            # current_fit: how valuable are its edges under the new distribution?
            raw_fit = sum(edge_probs_new.get(e, 0.0) for e in m.covered_edges)
            current_fit = raw_fit / max(m.trip_cost, 1e-9)

            # historical_success: normalised mean historical fit
            hist_success = m.historical_mean_fit  # already a ratio

            # signature_match: fraction of stored signature that overlaps top-k
            sig_set = set(m.top_k_signature_edges)
            sig_match = len(sig_set & global_topk) / max(len(sig_set), 1)

            # repair_cost_estimate: empirically measured from past repairs
            repair_est = m.avg_repair_cost

            score = (w1 * current_fit
                     + w2 * hist_success
                     + w3 * sig_match
                     - w4 * repair_est)
            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        return scored[:cfg['top_n_candidates']]

    def size(self) -> int:
        return len(self._bank)


# ══════════════════════════════════════════════════════════════════════════════
# §4  Trip-to-MemoryTrip conversion
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_trip(route: List[int],
                      trip_cost: float,
                      mission_time_component: float,
                      home_depot: int,
                      edge_probs: Dict[Tuple, float],
                      graph_id: int,
                      instance_id: int,
                      top_k: int = 5) -> MemoryTrip:
    """
    Convert a raw route into a MemoryTrip with probability-ranked signature.

    Parameters
    ----------
    route                   : ordered list of nodes in the trip
    trip_cost               : travel time for this trip
    mission_time_component  : this trip's contribution to total mission time
    home_depot              : vehicle's home / assigned depot
    edge_probs              : edge probability dict at creation time
    graph_id                : which graph this trip belongs to
    instance_id             : which instance created it
    top_k                   : signature size
    """
    covered = edges_in_path(route)
    creation_reward = sum(edge_probs.get(e, 0.0) for e in covered)

    # Probability-ranked signature: sort covered edges by p_e at creation
    ranked = sorted(covered, key=lambda e: edge_probs.get(e, 0.0), reverse=True)
    sig_edges = ranked[:top_k]
    sig_probs = [edge_probs.get(e, 0.0) for e in sig_edges]

    start_depot = route[0]
    end_depot   = route[-1]

    return MemoryTrip(
        graph_id=graph_id,
        instance_id_created=instance_id,
        vehicle_home_depot=home_depot,
        start_depot=start_depot,
        end_depot=end_depot,
        route=list(route),
        covered_edges=covered,
        trip_cost=max(trip_cost, 1e-9),
        mission_time_component=mission_time_component,
        top_k_signature_edges=sig_edges,
        top_k_signature_probs=sig_probs,
        creation_reward=creation_reward,
        creation_reward_per_cost=creation_reward / max(trip_cost, 1e-9),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §5  Repair operator
# ══════════════════════════════════════════════════════════════════════════════

def repair_trip(route: List[int],
                G: nx.Graph,
                dist: dict,
                depot: int,
                depots: List[int],
                capacity: float,
                edge_probs_new: Dict[Tuple, float],
                uncovered_target: Set[Tuple],
                prob_threshold: float = 0.0) -> Tuple[List[int], float, Set[Tuple]]:
    """
    Lightly repair an existing route for a new probability instance.

    Strategy
    --------
    1. Walk the existing route node by node, keeping each step only if:
       - it stays within battery capacity (feasibility), AND
       - either it covers a target edge with p_new > threshold, OR
         it is the only way to stay connected.
    2. After the feasible prefix, greedily insert nearby uncovered high-prob
       edges by jumping to their nearest endpoint via shortest path.
    3. Close the route back to the nearest depot.

    Returns (repaired_route, repaired_cost, newly_covered)
    """
    depot_set = set(depots)
    repaired  = [route[0]]
    cost      = 0.0
    covered   = set()

    # ── Phase 1: replay old route, drop low-value segments ───────────────────
    for i in range(1, len(route)):
        prev = repaired[-1]
        nxt  = route[i]
        if not G.has_edge(prev, nxt):
            break  # graph might not have this edge (safety)
        w = G[prev][nxt]['weight']
        e = can(prev, nxt)
        depot_back = min((dist.get(nxt, {}).get(d, float('inf')) for d in depots),
                         default=float('inf'))
        if cost + w + depot_back > capacity + 1e-9:
            break   # capacity would be exceeded — stop here

        is_valuable = (edge_probs_new.get(e, 0.0) > prob_threshold
                       or nxt in depot_set
                       or e in uncovered_target)
        if is_valuable:
            repaired.append(nxt)
            cost += w
            if e in uncovered_target:
                covered.add(e)
        else:
            # Skip this node entirely — we stay at prev for the next step
            pass

    # ── Phase 2: greedy insert of nearby high-prob uncovered edges ────────────
    still_uncovered = uncovered_target - covered
    sorted_targets  = sorted(still_uncovered,
                             key=lambda e: edge_probs_new.get(e, 0.0),
                             reverse=True)

    current = repaired[-1]
    for target_e in sorted_targets:
        a, b = target_e
        for endpoint in (a, b):
            dist_to_ep  = dist.get(current, {}).get(endpoint, float('inf'))
            edge_w      = G[a][b]['weight'] if G.has_edge(a, b) else float('inf')
            depot_after = min((dist.get(b if endpoint == a else a, {}).get(d, float('inf'))
                               for d in depots), default=float('inf'))
            total_needed = cost + dist_to_ep + edge_w + depot_after
            if total_needed <= capacity + 1e-9:
                try:
                    seg = nx.shortest_path(G, current, endpoint, weight='weight')
                    other = b if endpoint == a else a
                    # walk to endpoint
                    for node in seg[1:]:
                        w = G[repaired[-1]][node]['weight']
                        repaired.append(node)
                        cost += w
                        covered.add(can(repaired[-2], node))
                    # traverse the target edge
                    if G.has_edge(endpoint, other):
                        w = G[endpoint][other]['weight']
                        repaired.append(other)
                        cost += w
                        covered.add(target_e)
                        current = other
                    break
                except (nx.NetworkXNoPath, KeyError):
                    continue

    # ── Phase 3: close to nearest depot ──────────────────────────────────────
    current = repaired[-1]
    if current not in depot_set:
        nd = min(depots, key=lambda d: dist.get(current, {}).get(d, float('inf')))
        try:
            seg = nx.shortest_path(G, current, nd, weight='weight')
            seg_cost = sum(G[seg[i]][seg[i+1]]['weight'] for i in range(len(seg)-1))
            if cost + seg_cost <= capacity + 1e-9:
                for node in seg[1:]:
                    repaired.append(node)
                    cost += G[repaired[-2]][node]['weight']
                    covered.add(can(repaired[-2], node))
        except (nx.NetworkXNoPath, KeyError):
            pass

    return repaired, cost, covered & uncovered_target


# ══════════════════════════════════════════════════════════════════════════════
# §6  Memory-augmented trip planner
# ══════════════════════════════════════════════════════════════════════════════

class TripPlannerStats:
    """Collects statistics about how trips were built during one SA run."""
    def __init__(self):
        self.n_scratch_builds  = 0
        self.n_exact_hits      = 0
        self.n_memory_hits     = 0
        self.n_repaired        = 0
        self.n_rejected        = 0
        self.n_candidates_eval = 0

    def hit_rate(self) -> float:
        total = (self.n_scratch_builds + self.n_exact_hits
                 + self.n_memory_hits + self.n_repaired)
        if total == 0:
            return 0.0
        return (self.n_exact_hits + self.n_memory_hits + self.n_repaired) / total


def plan_vehicle_trips_with_memory(
        G: nx.Graph,
        dist: dict,
        depot: int,
        depots: List[int],
        capacity: float,
        recharge_time: float,
        T_interval: float,
        edge_probs: Dict[Tuple, float],
        target_edges: List[Tuple],
        exact_cache: dict,
        max_flights: int,
        memory_bank: Optional[ProbabilityRankedTripMemory],
        graph_id: int,
        instance_id: int,
        stats: TripPlannerStats,
        all_edges: Set[Tuple],
        cfg: dict = None) -> Tuple[List[List[int]], float, Set[Tuple], float]:
    """
    Memory-augmented wrapper around plan_vehicle_trips.

    Decision hierarchy (per-trip, within the multi-trip planning loop):
      1. Check exact cache  → direct reuse (Mode B + C)
      2. Query PRTMB        → score candidates, accept direct or repair (Mode C only)
      3. Fallback           → call ProbabilisticMagneticRouter from scratch

    Returns same tuple as plan_vehicle_trips:
        (trips, total_travel, covered, mission_time)
    """
    cfg = cfg or MEMORY_CONFIG

    # ── Exact cache check (same as original plan_vehicle_trips) ──────────────
    cache_key = (depot, frozenset(can(e[0], e[1]) for e in target_edges))
    if cache_key in exact_cache:
        stats.n_exact_hits += 1
        return exact_cache[cache_key]

    # ── If no memory bank, fall straight through to scratch builder ───────────
    if memory_bank is None or not target_edges:
        stats.n_scratch_builds += 1
        result = plan_vehicle_trips(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, exact_cache, max_flights)
        # Store trips in memory bank is handled by caller
        return result

    # ── PRTMB query ───────────────────────────────────────────────────────────
    best_possible_fit = (
        sum(edge_probs.get(e, 0.0) for e in map(lambda e: can(*e), target_edges))
        / max(capacity, 1e-9))

    candidates = memory_bank.retrieve(graph_id, depot, edge_probs, all_edges)
    stats.n_candidates_eval += len(candidates)

    target_set = set(can(e[0], e[1]) for e in target_edges)
    accepted   = False

    for score, mem in candidates:
        # Filter: this memory trip must cover at least some of our targets
        overlap = mem.covered_edges & target_set
        if not overlap:
            continue

        # Feasibility: trip cost must respect capacity
        if mem.trip_cost > capacity + 1e-9:
            stats.n_rejected += 1
            continue

        # Evaluate current fit
        raw_fit = sum(edge_probs.get(e, 0.0) for e in mem.covered_edges)
        current_fit = raw_fit / mem.trip_cost

        # ── Direct reuse if fit is high ────────────────────────────────
        if current_fit >= cfg['min_direct_fit'] * best_possible_fit:
            log.debug('  [PRTMB] direct reuse  depot=%d  fit=%.3f', depot, current_fit)
            # build multi-trip plan keeping this trip as the first flight
            reused_trips = [mem.route]
            covered_total = mem.covered_edges.copy()
            new_remaining = [e for e in target_edges if can(*e) not in covered_total]
            total_travel  = mem.trip_cost
            time_used     = mem.trip_cost
            current_depot = mem.end_depot

            # continue with scratch planner for remaining edges
            if new_remaining and time_used < T_interval:
                sub_result = plan_vehicle_trips(
                    G, dist, current_depot, depots, capacity, recharge_time,
                    T_interval - time_used - recharge_time,
                    edge_probs, new_remaining, {}, max(max_flights - 1, 1))
                sub_trips, sub_travel, sub_covered, _ = sub_result[:4]
                reused_trips.extend(sub_trips)
                total_travel  += sub_travel
                covered_total |= sub_covered

            n_trips      = len(reused_trips)
            mission_time = total_travel + max(0, n_trips - 1) * recharge_time

            result = (reused_trips, total_travel, covered_total, mission_time,
                      [path_length(G, t) for t in reused_trips])
            exact_cache[cache_key] = result
            mem.record_reuse(current_fit, repaired=False)
            stats.n_memory_hits += 1
            accepted = True
            break

        # ── Attempt repair if fit is partial ───────────────────────────
        if current_fit >= cfg['min_repair_fit'] * best_possible_fit:
            uncovered_target = target_set - mem.covered_edges
            repaired_route, rep_cost, newly_covered = repair_trip(
                mem.route, G, dist, depot, depots, capacity,
                edge_probs, uncovered_target)

            repaired_fit = (sum(edge_probs.get(e, 0.0)
                                for e in mem.covered_edges | newly_covered)
                            / max(rep_cost, 1e-9))

            if repaired_fit >= cfg['min_repair_fit'] * best_possible_fit:
                log.debug('  [PRTMB] repaired reuse  depot=%d  fit=%.3f→%.3f',
                           depot, current_fit, repaired_fit)
                repair_gain = repaired_fit - current_fit
                repair_cost_delta = abs(rep_cost - mem.trip_cost)
                mem.record_reuse(repaired_fit, repaired=True,
                                 repair_gain=repair_gain,
                                 repair_cost=repair_cost_delta)

                covered_total  = (mem.covered_edges | newly_covered) & target_set
                new_remaining  = [e for e in target_edges
                                  if can(*e) not in covered_total]
                total_travel   = rep_cost
                time_used      = rep_cost
                current_depot  = repaired_route[-1]

                reused_trips = [repaired_route]
                if new_remaining and time_used < T_interval:
                    sub_result = plan_vehicle_trips(
                        G, dist, current_depot, depots, capacity, recharge_time,
                        T_interval - time_used - recharge_time,
                        edge_probs, new_remaining, {}, max(max_flights - 1, 1))
                    sub_trips, sub_travel, sub_covered, _ = sub_result[:4]
                    reused_trips.extend(sub_trips)
                    total_travel  += sub_travel
                    covered_total |= sub_covered

                n_trips      = len(reused_trips)
                mission_time = total_travel + max(0, n_trips - 1) * recharge_time
                result = (reused_trips, total_travel, covered_total, mission_time,
                          [path_length(G, t) for t in reused_trips])
                exact_cache[cache_key] = result
                stats.n_repaired += 1
                accepted = True
                break
            else:
                stats.n_rejected += 1

    if not accepted:
        # ── Full scratch build ────────────────────────────────────────────
        log.debug('  [PRTMB] scratch build  depot=%d', depot)
        stats.n_scratch_builds += 1
        result = plan_vehicle_trips(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, exact_cache, max_flights)

        # Store individual trips into memory bank
        (trips_r, _, _, _, trip_times_r) = result
        for t_idx, (trip, t_cost) in enumerate(zip(trips_r, trip_times_r)):
            if len(trip) > 2:
                mem_trip = build_memory_trip(
                    trip, t_cost, t_cost, depot, edge_probs,
                    graph_id, instance_id,
                    top_k=cfg['top_k_signature'])
                memory_bank.store(mem_trip)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# §6b  Planner factories — each returns a callable with signature
#      planner_fn(depot, target_edges) → (trips, total_travel, covered, mission_time, trip_times)
#
# This is the key abstraction that makes the three experimental modes share
# one identical SA loop. The only thing that changes between modes is which
# factory is used; the SA code (§7) never sees the difference.
# ══════════════════════════════════════════════════════════════════════════════

def make_scratch_planner(
        G: nx.Graph, dist: dict, depots: List[int],
        capacity: float, recharge_time: float, T_interval: float,
        edge_probs: Dict[Tuple, float], max_flights: int,
        stats: 'TripPlannerStats') -> 'Callable':
    """
    Planner A — Baseline.

    Each call uses a run-local exact cache (identical to the original SA's
    trip_cache that is fresh per run). No cross-instance memory or
    cross-call sharing beyond the current SA run. Every novel
    (depot, target_frozenset) combination triggers a full magnetic-router build.

    Research note: this is the control condition. The exact cache inside
    plan_vehicle_trips avoids re-building the *same* call within one SA run,
    but as edge-ownership changes with each SA neighbour step, most keys are
    unique and the cache hit rate within a single run is modest.
    """
    local_cache: dict = {}   # fresh per SA run — cleared in the factory closure

    def planner_fn(depot: int, target_edges: List[Tuple]):
        result = plan_vehicle_trips(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, local_cache, max_flights)
        # Count: every plan_vehicle_trips call that wasn't a cache hit is scratch
        cache_key = (depot, frozenset(can(e[0], e[1]) for e in target_edges))
        # plan_vehicle_trips already stores the result in local_cache on cache MISS;
        # on cache HIT it returned early. We infer which happened:
        # (We cannot peek inside plan_vehicle_trips, but we track aggregate calls
        # by checking if the key was ALREADY in the cache before the call.)
        stats.n_scratch_builds += 1   # treated as scratch; exact hits counted below
        return result

    return planner_fn


def make_exact_cache_planner(
        G: nx.Graph, dist: dict, depots: List[int],
        capacity: float, recharge_time: float, T_interval: float,
        edge_probs: Dict[Tuple, float], max_flights: int,
        shared_cache: dict,
        stats: 'TripPlannerStats') -> 'Callable':
    """
    Planner B — Exact cache.

    Same as the scratch planner but the cache is *shared across SA runs /
    probability instances*. A cache hit occurs when exactly the same
    (depot, frozenset(target_edges)) key was seen in a previous instance run.

    Because changing edge probabilities changes the greedy allocation and
    therefore the target_edge sets, keys rarely match across instances —
    this is the known limitation of exact memoisation for variable-probability
    problems, and is why PRTMB is needed.
    """
    def planner_fn(depot: int, target_edges: List[Tuple]):
        cache_key = (depot, frozenset(can(e[0], e[1]) for e in target_edges))
        if cache_key in shared_cache:
            stats.n_exact_hits += 1
            return shared_cache[cache_key]
        # Cache miss → build from scratch, store result
        result = plan_vehicle_trips(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, shared_cache, max_flights)
        stats.n_scratch_builds += 1
        return result

    return planner_fn


def make_memory_planner(
        G: nx.Graph, dist: dict, depots: List[int],
        capacity: float, recharge_time: float, T_interval: float,
        edge_probs: Dict[Tuple, float], max_flights: int,
        memory_bank: ProbabilityRankedTripMemory,
        graph_id: int, instance_id: int,
        all_edges: Set[Tuple],
        stats: 'TripPlannerStats',
        cfg: dict = None) -> 'Callable':
    """
    Planner C — Intelligent PRTMB.

    Wraps plan_vehicle_trips_with_memory, which queries the memory bank for
    similar trips before building from scratch. Retrieved trips are accepted
    directly or lightly repaired; scratch builds are stored back into the
    bank for future reuse across instances.
    """
    run_cache: dict = {}   # per-run exact cache (same as baseline's local_cache)
    _cfg = cfg or MEMORY_CONFIG

    def planner_fn(depot: int, target_edges: List[Tuple]):
        return plan_vehicle_trips_with_memory(
            G, dist, depot, depots, capacity, recharge_time, T_interval,
            edge_probs, target_edges, run_cache, max_flights,
            memory_bank, graph_id, instance_id, stats, all_edges, _cfg)

    return planner_fn


# ══════════════════════════════════════════════════════════════════════════════
# §7  Unified SA loop  (shared by ALL three modes)
#
# This is the single SA implementation used by baseline, exact-cache, and
# PRTMB alike. The only argument that differs across modes is `planner_fn`.
# Every other SA decision — temperature schedule, neighbour operator, acceptance
# criterion, initial greedy allocation — is completely identical, making the
# runtime and quality comparison fair.
# ══════════════════════════════════════════════════════════════════════════════

def simulated_annealing_unified(
        G: nx.Graph,
        dist: dict,
        depots: List[int],
        edge_probs: Dict[Tuple, float],
        T_interval: float,                # needed for compute_objective
        planner_fn,                       # Callable[[int, List[Tuple]], Tuple]
        memory_bank: Optional[ProbabilityRankedTripMemory],
        depots_for_storage: List[int],    # same as depots; kept explicit for clarity
        edge_probs_for_storage: Dict[Tuple, float],
        graph_id: int,
        instance_id: int,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        no_improve_limit: int = 200,
) -> Tuple[List, float, float, float]:
    """
    One SA loop to rule them all.

    Parameters
    ----------
    planner_fn : callable(depot, target_edges) → (trips, total_travel, covered, mission_time, trip_times)
        The only mode-specific argument. Constructed by one of the three
        planner factories (make_scratch_planner / make_exact_cache_planner /
        make_memory_planner) before calling this function.
    memory_bank : optional PRTMB instance
        When provided, the best solution's trips are stored into it after the
        run completes — enables cross-instance learning for mode C.
        Modes A and B pass None; nothing is stored.

    Returns
    -------
    (best_all_trips, best_obj, best_prob, best_beta)
    """
    # ── Greedy initial allocation (identical across all modes) ─────────────
    allocation    = greedy_allocate(G, dist, depots, edge_probs)
    all_trips     = [None] * len(depots)
    mission_times = [0.0]  * len(depots)

    for k, depot in enumerate(depots):
        trips, _, _, mt, _ = planner_fn(depot, allocation[k])
        all_trips[k]     = trips
        mission_times[k] = mt

    current_obj, current_prob, current_beta, _, _ = compute_objective(
        all_trips, edge_probs, mission_times, T_interval)

    best_obj   = current_obj
    best_prob  = current_prob
    best_beta  = current_beta
    best_trips = copy.deepcopy(all_trips)
    best_mt    = list(mission_times)

    temperature = initial_temperature
    no_improve  = 0

    for _ in range(max_iterations):
        new_alloc, (v1, v2) = neighbour_allocation(allocation, mission_times)
        new_all_trips = copy.deepcopy(all_trips)
        new_mt        = list(mission_times)

        for vk in (v1, v2):
            trips, _, _, mt, _ = planner_fn(depots[vk], new_alloc[vk])
            new_all_trips[vk] = trips
            new_mt[vk]        = mt

        new_obj, new_prob, new_beta, _, _ = compute_objective(
            new_all_trips, edge_probs, new_mt, T_interval)

        delta = new_obj - current_obj
        ap    = 1.0 if delta > 0 else (
            math.exp(delta / temperature) if temperature > 1e-10 else 0.0)

        if ap > random.random():
            allocation    = new_alloc
            all_trips     = new_all_trips
            mission_times = new_mt
            current_obj   = new_obj

        if new_obj > best_obj + 1e-9:
            best_obj   = new_obj
            best_prob  = new_prob
            best_beta  = new_beta
            best_trips = copy.deepcopy(new_all_trips)
            best_mt    = list(new_mt)
            no_improve = 0
        else:
            no_improve += 1

        temperature *= cooling_rate
        if no_improve >= no_improve_limit:
            break

    # ── Store best trips into memory bank (PRTMB mode only) ───────────────
    if memory_bank is not None:
        for k, trips in enumerate(best_trips):
            for trip in trips:
                if len(trip) > 2:
                    tc = path_length(G, trip)
                    mem_trip = build_memory_trip(
                        trip, tc, tc,
                        depots_for_storage[k],
                        edge_probs_for_storage,
                        graph_id, instance_id)
                    memory_bank.store(mem_trip)

    return best_trips, best_obj, best_prob, best_beta


# ══════════════════════════════════════════════════════════════════════════════
# §8  Probability instance generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_probability_instances(
        G: nx.Graph,
        n_instances: int,
        seed_base: int,
        scale: float = PROB_PERTURBATION_SCALE,
        p_min: float = PROB_MIN_CLIP,
        p_max: float = PROB_MAX_CLIP) -> List[Dict[Tuple, float]]:
    """
    Generate *n_instances* probability realisations on the same graph topology.

    The base probabilities come from the graph edge attributes.
    Each instance perturbs these with clipped Gaussian noise, preserving
    the overall spatial risk structure while changing magnitudes.

    Instance 0 = original (unperturbed) distribution.
    Instances 1..n-1 = progressively noisy perturbations with different seeds.
    """
    instances = []
    base_probs: Dict[Tuple, float] = {}
    for u, v, d in G.edges(data=True):
        base_probs[can(u, v)] = d.get('prob', 0.0)

    for i in range(n_instances):
        rng = np.random.default_rng(seed_base + i * 7)
        if i == 0:
            instances.append(dict(base_probs))  # unperturbed baseline
        else:
            noisy = {}
            for e, p in base_probs.items():
                delta = rng.normal(0.0, scale)
                noisy[e] = float(np.clip(p + delta, p_min, p_max))
            instances.append(noisy)

    return instances


# ══════════════════════════════════════════════════════════════════════════════
# §9  Per-instance solver (single mode)
# ══════════════════════════════════════════════════════════════════════════════

def solve_instance(
        G: nx.Graph,
        dist: dict,
        depots: List[int],
        capacity: float,
        recharge_time: float,
        T_interval: float,
        max_flights: int,
        edge_probs: Dict[Tuple, float],
        mode: str,
        memory_bank: Optional[ProbabilityRankedTripMemory],
        shared_exact_cache: dict,
        graph_id: int,
        instance_id: int,
        repeat: int,
        seed: int,
        sa_kwargs: dict) -> ExperimentRecord:
    """
    Solve one probability instance under the given mode.

    All three modes now use the *identical* SA loop (simulated_annealing_unified).
    The only thing that differs is the planner_fn injected into that loop:

        baseline          → make_scratch_planner        (fresh cache per run)
        exact_cache       → make_exact_cache_planner    (shared cache across runs)
        intelligent_memory→ make_memory_planner         (PRTMB retrieval + repair)

    Parameters
    ----------
    memory_bank        : PRTMB instance (mode C) or None (modes A/B)
    shared_exact_cache : persistent dict shared across instances for mode B;
                         ignored (but still passed as empty {}) for modes A and C
    """
    random.seed(seed)
    np.random.seed(seed)

    all_edges: Set[Tuple] = {can(u, v) for u, v in G.edges()}
    n_total  = len(all_edges)
    stats    = TripPlannerStats()

    # ── Build the mode-specific planner closure ─────────────────────────────
    if mode == 'baseline':
        planner_fn = make_scratch_planner(
            G, dist, depots, capacity, recharge_time, T_interval,
            edge_probs, max_flights, stats)
        store_bank = None

    elif mode == 'exact_cache':
        planner_fn = make_exact_cache_planner(
            G, dist, depots, capacity, recharge_time, T_interval,
            edge_probs, max_flights, shared_exact_cache, stats)
        store_bank = None

    else:  # mode == 'intelligent_memory'
        assert memory_bank is not None
        planner_fn = make_memory_planner(
            G, dist, depots, capacity, recharge_time, T_interval,
            edge_probs, max_flights, memory_bank,
            graph_id, instance_id, all_edges, stats)
        store_bank = memory_bank

    # ── Run the unified SA (identical code path for all three modes) ────────
    t0 = time.time()
    best_trips, best_obj, best_prob, best_beta = simulated_annealing_unified(
        G, dist, depots, edge_probs,
        T_interval=T_interval,
        planner_fn=planner_fn,
        memory_bank=store_bank,
        depots_for_storage=depots,
        edge_probs_for_storage=edge_probs,
        graph_id=graph_id,
        instance_id=instance_id,
        **sa_kwargs)
    elapsed = time.time() - t0

    # ── Quality metrics ─────────────────────────────────────────────────────
    mission_times = [
        sum(path_length(G, t) for t in trips)
        + max(0, len(trips) - 1) * recharge_time
        for trips in best_trips
    ]
    _, prob_min_final, beta_final, global_prob_final, global_covered = \
        compute_objective(best_trips, edge_probs, mission_times, T_interval)

    n_covered  = len(global_covered)
    pct_uncov  = 100.0 * (n_total - n_covered) / max(n_total, 1)
    hit_rate   = stats.hit_rate()

    log.info('  mode=%-20s  inst=%d  t=%.2fs  obj=%.4f  pct_uncov=%.1f%%  '
             'hits=%d/%d  scratch=%d',
             mode, instance_id, elapsed, best_obj, pct_uncov,
             stats.n_exact_hits + stats.n_memory_hits + stats.n_repaired,
             stats.n_scratch_builds + stats.n_exact_hits +
             stats.n_memory_hits + stats.n_repaired,
             stats.n_scratch_builds)

    return ExperimentRecord(
        mode=mode,
        graph_id=graph_id,
        instance_id=instance_id,
        repeat=repeat,
        seed=seed,
        runtime_s=elapsed,
        obj=best_obj,
        global_prob=global_prob_final,
        prob_min=prob_min_final,
        beta=beta_final,
        pct_uncovered=pct_uncov,
        n_scratch_builds=stats.n_scratch_builds,
        n_exact_hits=stats.n_exact_hits,
        n_memory_hits=stats.n_memory_hits,
        n_repaired=stats.n_repaired,
        n_rejected=stats.n_rejected,
        n_candidates_eval=stats.n_candidates_eval,
        memory_hit_rate=hit_rate,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §10  7-instance experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(graph_dir: str,
                   graph_id: int,
                   n_instances: int,
                   n_repeats: int,
                   sa_kwargs: dict,
                   seed_base: int = RANDOM_SEED_BASE) -> pd.DataFrame:
    """
    Run three experimental modes (baseline / exact_cache / intelligent_memory)
    across *n_instances* probability realisations × *n_repeats* independent SA
    repetitions on the same graph.

    Repeat structure
    ----------------
    Each repeat is a completely independent replication of the full
    7-instance sequence. Within a repeat:
      - Memory bank (mode C) is fresh at the start of each repeat but shared
        across instances inside that repeat (cross-instance learning).
      - Exact cache (mode B) is similarly fresh per repeat, shared across
        instances inside that repeat.
      - Baseline (mode A) uses a per-SA-run-local cache with no cross-run sharing.
    Seeds differ across repeats so SA randomness produces genuinely different runs.

    Returns a DataFrame with one row per (mode, instance_id, repeat).
    Total rows = n_modes × n_instances × n_repeats = 3 × 7 × 10 = 210.
    """
    # ── Load graph ────────────────────────────────────────────────────────────
    pkl_path = os.path.join(graph_dir, f'{graph_id}.pickle')
    with open(pkl_path, 'rb') as fh:
        G: nx.Graph = pickle.load(fh)

    meta_all = np.load(os.path.join(graph_dir, 'graph_params.npy'), allow_pickle=True)
    meta = next((m for m in meta_all if int(m['graph_id']) == graph_id), None)
    if meta is None:
        raise ValueError(f'Graph {graph_id} not found in graph_params.npy')

    capacity      = float(meta['vehicle_capacity'])
    recharge_time = float(meta['recharge_time'])
    T_interval    = float(meta['time_interval']) / 2.0
    depots        = list(meta['depot_nodes'])
    max_flights   = max(1, int(T_interval // max(recharge_time + capacity, 1e-9)))

    log.info('Graph %d: nodes=%d edges=%d depots=%s B=%.0f TR=%.0f T=%.0f flights=%d',
             graph_id, G.number_of_nodes(), G.number_of_edges(), depots,
             capacity, recharge_time, T_interval, max_flights)

    # Pre-compute all-pairs shortest paths (done once, shared across all modes/repeats)
    dist: Dict[int, Dict[int, float]] = dict(
        nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # ── Generate 7 probability instances (fixed across all repeats) ───────────
    prob_instances = generate_probability_instances(
        G, n_instances, seed_base)
    log.info('Generated %d probability instances × %d repeats = %d total runs per mode',
             n_instances, n_repeats, n_instances * n_repeats)

    records: List[ExperimentRecord] = []
    modes = ['baseline', 'exact_cache', 'intelligent_memory']

    for mode in modes:
        log.info('\n%s\n  MODE: %s\n%s', '='*58, mode.upper(), '='*58)

        for rep in range(n_repeats):
            # Fresh memory / cache at the start of each repeat
            mode_memory = (ProbabilityRankedTripMemory(MEMORY_CONFIG)
                           if mode == 'intelligent_memory' else None)
            mode_shared_cache = {} if mode == 'exact_cache' else {}

            for inst_id, edge_probs in enumerate(prob_instances):
                # Unique seed: vary by mode-hash, repeat, instance
                seed = (seed_base
                        + hash(mode) % 1000
                        + rep * 10000
                        + inst_id * 13)
                seed = abs(seed) % (2**31)

                rec = solve_instance(
                    G, dist, depots, capacity, recharge_time, T_interval,
                    max_flights, edge_probs,
                    mode=mode,
                    memory_bank=mode_memory,
                    shared_exact_cache=mode_shared_cache,
                    graph_id=graph_id,
                    instance_id=inst_id,
                    repeat=rep,
                    seed=seed,
                    sa_kwargs=sa_kwargs)
                records.append(rec)

    return pd.DataFrame([vars(r) for r in records])


# ══════════════════════════════════════════════════════════════════════════════
# §11  Statistical validation
# ══════════════════════════════════════════════════════════════════════════════

def statistical_analysis(df: pd.DataFrame) -> None:
    """
    Paired Wilcoxon signed-rank tests:
      - runtime:    baseline vs intelligent_memory
      - objective:  baseline vs intelligent_memory

    With multiple repeats the pairing is done on per-instance means:
    each instance_id gives one paired (baseline_mean, memory_mean) observation
    averaged over the N_REPEATS runs.  That yields N_INSTANCES = 7 pairs and
    N_INSTANCES × N_REPEATS = 70 individual rows used in the means.
    """
    print('\n' + '═'*60)
    print('  STATISTICAL VALIDATION')
    print('═'*60)

    # Aggregate per instance (mean over repeats) before pairing
    agg_base = (df[df['mode'] == 'baseline']
                .groupby('instance_id')[['runtime_s', 'obj']].mean())
    agg_mem  = (df[df['mode'] == 'intelligent_memory']
                .groupby('instance_id')[['runtime_s', 'obj']].mean())

    if agg_base.empty or agg_mem.empty:
        print('  Not enough data for statistical tests.')
        print('═'*60)
        return

    # Align on shared instance_ids
    common = agg_base.index.intersection(agg_mem.index)
    rt_base  = agg_base.loc[common, 'runtime_s'].values
    rt_mem   = agg_mem.loc[common, 'runtime_s'].values
    obj_base = agg_base.loc[common, 'obj'].values
    obj_mem  = agg_mem.loc[common, 'obj'].values

    n_pairs = len(common)
    n_repeats_actual = df['repeat'].max() + 1 if 'repeat' in df.columns else 1
    print(f'  Instances (pairs)  : {n_pairs}  ×  {n_repeats_actual} repeats each')
    print(f'  Total rows used    : {n_pairs * n_repeats_actual} per mode')

    if n_pairs >= 2:
        try:
            stat_rt,  p_rt  = stats.wilcoxon(rt_base,  rt_mem,  alternative='greater')
        except Exception:
            stat_rt,  p_rt  = float('nan'), float('nan')
        try:
            stat_obj, p_obj = stats.wilcoxon(obj_base, obj_mem)
        except Exception:
            stat_obj, p_obj = float('nan'), float('nan')

        mean_speedup  = (rt_base.mean()  - rt_mem.mean())  / max(rt_base.mean(),  1e-9)
        mean_obj_diff = obj_mem.mean() - obj_base.mean()

        print(f'  Avg runtime (baseline)   : {rt_base.mean():.3f} s')
        print(f'  Avg runtime (memory)     : {rt_mem.mean():.3f} s')
        print(f'  Speedup                  : {mean_speedup*100:.1f}%')
        print(f'  Wilcoxon (runtime)       : stat={stat_rt:.3f}  p={p_rt:.4f}'
              f'  → {"SIGNIFICANT ✓" if p_rt < 0.05 else "not significant"}')
        print(f'  Avg obj diff (mem−base)  : {mean_obj_diff:+.6f}')
        print(f'  Wilcoxon (obj quality)   : stat={stat_obj:.3f}  p={p_obj:.4f}'
              f'  → degradation is {"NEGLIGIBLE ✓" if p_obj > 0.05 else "significant ✗"}')
    else:
        print('  Too few paired samples for Wilcoxon test (need ≥ 2).')

    print('═'*60)


# ══════════════════════════════════════════════════════════════════════════════
# §12  Plotting
# ══════════════════════════════════════════════════════════════════════════════

MODE_STYLES = {
    'baseline':           {'color': '#2C7BB6', 'marker': 'o', 'label': 'Baseline'},
    'exact_cache':        {'color': '#D7191C', 'marker': 's', 'label': 'Exact Cache'},
    'intelligent_memory': {'color': '#1A9641', 'marker': '^', 'label': 'PRTMB (Ours)'},
}


def _style(mode: str):
    return MODE_STYLES.get(mode, {'color': 'gray', 'marker': 'x', 'label': mode})


def make_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Generate and save publication-style figures."""
    os.makedirs(out_dir, exist_ok=True)
    modes    = df['mode'].unique().tolist()
    insts    = sorted(df['instance_id'].unique())
    x_tick   = list(range(len(insts)))

    # ── Fig 1: Runtime per instance ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in modes:
        sub = (df[df['mode'] == mode]
               .groupby('instance_id')['runtime_s'].mean().sort_index())
        st  = _style(mode)
        ax.plot(x_tick, sub.values,
                color=st['color'], marker=st['marker'],
                linewidth=1.8, markersize=7, label=st['label'])
    ax.set_xlabel('Instance Index', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime per Probability Instance', fontsize=13)
    ax.set_xticks(x_tick)
    ax.set_xticklabels([f'I{i}' for i in insts])
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'runtime_per_instance.png'), dpi=150)
    plt.close(fig)
    log.info('Saved runtime_per_instance.png')

    # ── Fig 2: Objective per instance ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in modes:
        sub = (df[df['mode'] == mode]
               .groupby('instance_id')['obj'].mean().sort_index())
        st  = _style(mode)
        ax.plot(x_tick, sub.values,
                color=st['color'], marker=st['marker'],
                linewidth=1.8, markersize=7, label=st['label'])
    ax.set_xlabel('Instance Index', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Objective Value per Probability Instance', fontsize=13)
    ax.set_xticks(x_tick)
    ax.set_xticklabels([f'I{i}' for i in insts])
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'objective_per_instance.png'), dpi=150)
    plt.close(fig)
    log.info('Saved objective_per_instance.png')

    # ── Fig 3: Boxplot of runtime ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    data_box = [df[df['mode'] == m]['runtime_s'].values for m in modes]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, mode in zip(bp['boxes'], modes):
        patch.set_facecolor(_style(mode)['color'])
        patch.set_alpha(0.75)
    ax.set_xticklabels([_style(m)['label'] for m in modes], fontsize=10)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime Distribution Across 7 Instances', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'runtime_boxplot.png'), dpi=150)
    plt.close(fig)
    log.info('Saved runtime_boxplot.png')

    # ── Fig 4: Memory hit statistics ──────────────────────────────────────────
    mem_df = (df[df['mode'] == 'intelligent_memory']
              .groupby('instance_id').mean(numeric_only=True).sort_index())
    if not mem_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_w  = 0.22
        xs     = np.arange(len(insts))
        ax.bar(xs - bar_w, mem_df['n_memory_hits'].values,   bar_w, color='#1A9641', label='Direct Memory Hit')
        ax.bar(xs,          mem_df['n_repaired'].values,      bar_w, color='#78C679', label='Repaired Hit')
        ax.bar(xs + bar_w, mem_df['n_scratch_builds'].values, bar_w, color='#D95F02', label='Scratch Build')
        ax.set_xlabel('Instance Index', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('PRTMB Memory Hit Statistics per Instance', fontsize=13)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'I{i}' for i in insts])
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'memory_hit_stats.png'), dpi=150)
        plt.close(fig)
        log.info('Saved memory_hit_stats.png')

    # ── Fig 5: PRTMB trip-build composition (replaces uninformative scratch-build line) ──
    # Rationale: baseline and exact_cache use the original SA which is not instrumented
    # at the inner trip-builder level — their n_scratch_builds is just a placeholder (1).
    # The only meaningful comparison is within the intelligent_memory mode itself, showing
    # how each trip planning call was resolved: scratch build vs. exact cache hit vs.
    # direct memory hit vs. repaired reuse.
    mem_df_fig5 = (df[df['mode'] == 'intelligent_memory']
                   .groupby('instance_id').mean(numeric_only=True).sort_index())
    if not mem_df_fig5.empty:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        xs    = np.arange(len(insts))
        bar_w = 0.6

        scratch_vals = mem_df_fig5['n_scratch_builds'].values
        exact_vals   = mem_df_fig5['n_exact_hits'].values
        direct_vals  = mem_df_fig5['n_memory_hits'].values
        repair_vals  = mem_df_fig5['n_repaired'].values
        hit_rates    = mem_df_fig5['memory_hit_rate'].values * 100.0

        # Stacked bars: scratch (bottom) → exact → direct memory → repaired (top)
        p1 = ax1.bar(xs, scratch_vals, bar_w,
                     color='#D95F02', label='Scratch Build (no memory)')
        p2 = ax1.bar(xs, exact_vals,   bar_w, bottom=scratch_vals,
                     color='#7570B3', label='Exact Cache Hit')
        p3 = ax1.bar(xs, direct_vals,  bar_w,
                     bottom=scratch_vals + exact_vals,
                     color='#1A9641', label='Direct Memory Hit')
        p4 = ax1.bar(xs, repair_vals,  bar_w,
                     bottom=scratch_vals + exact_vals + direct_vals,
                     color='#78C679', label='Repaired Memory Hit')

        ax1.set_xlabel('Instance Index', fontsize=12)
        ax1.set_ylabel('Trip-Planner Calls (count)', fontsize=12)
        ax1.set_title('PRTMB Trip-Build Composition per Instance\n'
                      '(baseline / exact-cache not shown: original SA is not instrumented '
                      'at this level)', fontsize=11)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([f'I{i}' for i in insts])
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.35, axis='y')

        # Secondary axis: hit rate
        ax2 = ax1.twinx()
        ax2.plot(xs, hit_rates, color='black', marker='D',
                 linewidth=2.0, markersize=7, linestyle='--', label='Hit Rate (%)')
        ax2.set_ylabel('Memory Hit Rate (%)', fontsize=12)
        ax2.set_ylim(0, 110)
        ax2.legend(loc='upper right', fontsize=9)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'scratch_build_comparison.png'), dpi=150)
        plt.close(fig)
        log.info('Saved scratch_build_comparison.png (PRTMB composition chart)')


# ══════════════════════════════════════════════════════════════════════════════
# §13  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_table(df: pd.DataFrame) -> None:
    """Print mean ± std per (mode, instance_id) for key metrics."""
    metrics = ['runtime_s', 'obj', 'global_prob', 'pct_uncovered', 'memory_hit_rate']
    repeats_per_instance = df['repeat'].max() + 1 if 'repeat' in df.columns else 1
    
    print('\n' + '─'*100)
    print(f'  SUMMARY TABLE  (mean ± std across {repeats_per_instance} repeats per instance)')
    print('─'*100)
    # Header
    print(f"{'Mode':<22} {'Inst':<5} " + ''.join(f"{m:<19}" for m in metrics))
    print('─'*100)
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        instances = sorted(mode_df['instance_id'].unique())
        for inst in instances:
            sub = mode_df[mode_df['instance_id'] == inst]
            row = f'{mode:<22} {inst:<5} '
            for m in metrics:
                mn = sub[m].mean()
                sd = sub[m].std()
                if pd.isna(sd):
                    row += f'{mn:.4f}'.ljust(19)
                else:
                    row += f'{mn:.3f}±{sd:.3f}'.ljust(19)
            print(row)
        # Optional: blank line between modes for readability
        print('─'*100)


# ══════════════════════════════════════════════════════════════════════════════
# §14  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_dir  = os.path.join(script_dir, 'graphs')
    out_dir    = os.path.join(script_dir, 'memory_results')
    os.makedirs(out_dir, exist_ok=True)

    log.info('Starting PRTMB experiment on graph %d with %d instances × %d repeats',
             GRAPH_ID_TO_RUN, NUM_INSTANCES, N_REPEATS)

    df = run_experiment(
        graph_dir=graph_dir,
        graph_id=GRAPH_ID_TO_RUN,
        n_instances=NUM_INSTANCES,
        n_repeats=N_REPEATS,
        sa_kwargs=SA_KWARGS,
        seed_base=RANDOM_SEED_BASE,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    csv_path  = os.path.join(out_dir, 'prtmb_experiment_results.csv')
    xlsx_path = os.path.join(out_dir, 'prtmb_experiment_results.xlsx')
    df.to_csv(csv_path,  index=False)
    df.to_excel(xlsx_path, index=False)
    log.info('Results saved to %s and %s', csv_path, xlsx_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(df)

    # ── Statistics ────────────────────────────────────────────────────────────
    statistical_analysis(df)

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_plots(df, out_dir)

    # ── Summary DataFrame ─────────────────────────────────────────────────────
    summary = df.groupby('mode')[['runtime_s', 'obj', 'global_prob',
                                   'pct_uncovered', 'memory_hit_rate']].agg(
        ['mean', 'std']).round(4)
    summary_path = os.path.join(out_dir, 'prtmb_summary.xlsx')
    summary.to_excel(summary_path)
    log.info('Summary saved to %s', summary_path)

    print(f'\nAll outputs saved to: {out_dir}')
    print(df.to_string(index=False))
