#!/usr/bin/env python3


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
from magnetic_sa_mdrpp_rrv_cache_v2 import (
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
from magnetic_sa_mdrpp_rrv import simulated_annealing_magnetic_no_cache

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
    cooling_rate=0.99,
    max_iterations=500,
    no_improve_limit=300,
)

# Experiment configuration
NUM_INSTANCES   = 4
N_REPEATS       = 10   # independent SA repeats per (mode × instance) for mean±std
RANDOM_SEED_BASE = 42
PROB_PERTURBATION_SCALE = 0.25   # Gaussian σ for probability perturbation
PROB_MIN_CLIP = 0.05
PROB_MAX_CLIP = 0.95
GRAPH_ID_TO_RUN = 1              # which graph to use for the 7-instance experiment


# Removed generate_probability_instances since edge probabilities are loaded dynamically from NPY.


# ══════════════════════════════════════════════════════════════════════════════
# §9  Memory Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    start_depot: int
    end_depot: int
    route: Tuple[int, ...]
    covered: frozenset
    trip_cost: float
    current_reward: float = 0.0
    hist_avg_reward: float = 0.0
    hist_best_reward: float = 0.0
    periods_seen: int = 1
    periods_accepted: int = 0
    last_seeded_period: int = 0
    last_accepted_period: int = 0
    last_merged_period: int = 0

    @property
    def utility(self) -> float:
        """
        Ranking score for long-term memory entries.
        Utility is based on:
        - current-period reward density (heavy weight)
        - historical average reward density
        - true acceptance rate across SA evaluations
        """
        if self.trip_cost <= 1e-9:
            return 0.0
            
        cur_density = self.current_reward / self.trip_cost
        hist_density = self.hist_avg_reward / self.trip_cost
        
        # acceptance rate (true SA accepted count)
        acc_rate = self.periods_accepted / max(1, self.periods_seen)
        
        # Weighted utility
        return (cur_density * 0.5) + (hist_density * 0.3) + (acc_rate * 0.2)


class LongTermMemoryBank:
    """
    Cross-period structural memory bank.
    Stores robust depot-to-depot trips evaluated and promoted over multiple periods.
    """
    def __init__(self, max_per_depot=100):
        self.max_per_depot = max_per_depot
        
        # start_depot -> { (end_depot, covered_edges) : MemoryEntry }
        self.entries: Dict[int, Dict[Tuple[int, frozenset], MemoryEntry]] = {}
        
        self.total_promoted = 0
        self.total_pruned = 0

    def rescore_and_select(self, current_period: int, current_probs: Dict[Tuple, float], top_k=50) -> Dict[int, List[MemoryEntry]]:
        """
        At the start of a period:
        1. Recompute rewards for all memory entries using new probabilities.
        2. Rank by utility.
        3. Select top-K per depot to seed the short-term cache.
        """
        selected_entries = {}
        
        for depot, cache in self.entries.items():
            for (end_depot, cov), entry in cache.items():
                reward = sum(current_probs.get(e, 0.0) for e in cov)
                entry.current_reward = reward
                
            # Sort by utility descending
            ranked = sorted(cache.values(), key=lambda x: x.utility, reverse=True)
            top_entries = ranked[:top_k]
            
            for e in top_entries:
                e.last_seeded_period = current_period
                
            selected_entries[depot] = top_entries
            
        return selected_entries

    def seed_trip_cache(self, trip_cache, selected_entries: Dict[int, List[MemoryEntry]], all_depots: Set[int]):
        """
        Insert the selected short-list into the fresh per-period TripCache.
        """
        seeded_count = 0
        for depot, entries in selected_entries.items():
            for entry in entries:
                # Direct injection
                trip_cache.add_trip(
                    entry.start_depot,
                    list(entry.route),
                    entry.trip_cost,
                    set(entry.covered),
                    all_depots,
                    entry.current_reward
                )
                seeded_count += 1
        return seeded_count

    def merge_trips(self, current_period: int, best_trips: List[List[List[int]]], current_probs: Dict[Tuple, float], G: nx.Graph):
        """
        At the end of a period:
        Merge useful trips from the best SA solution back into the long-term bank.
        """
        for vehicle_trips in best_trips:
            for route_idx, route in enumerate(vehicle_trips):
                start_depot = route[0]
                end_depot = route[-1]
                cov = frozenset(edges_in_path(route))
                
                if not cov:
                    continue  # skip empty trips
                    
                cost = path_length(G, route)
                reward = sum(current_probs.get(e, 0.0) for e in cov)
                
                self._merge_single(start_depot, end_depot, tuple(route), cov, cost, reward, current_period)
                
                # Bidirectional storage if start and end depots are different
                # Note: valid depot check already done at SA routing logic level
                if start_depot != end_depot:
                    rev_route = tuple(reversed(route))
                    self._merge_single(end_depot, start_depot, rev_route, cov, cost, reward, current_period)

    def _merge_single(self, start_depot, end_depot, route, cov, cost, reward, current_period):
        if start_depot not in self.entries:
            self.entries[start_depot] = {}
            
        cache = self.entries[start_depot]
        new_key = (end_depot, cov)
        
        # 1. Check if strictly dominated by an existing entry
        for (e_end, e_cov), e_entry in cache.items():
            if e_end == end_depot and e_cov.issuperset(cov) and e_entry.trip_cost <= cost:
                if e_cov != cov or e_entry.trip_cost < cost:
                    return  # Strictly dominated, discard new

        # 2. Check if the new candidate dominates existing entries
        dominated_keys = []
        for (e_end, e_cov), e_entry in list(cache.items()):
            if e_end == end_depot and cov.issuperset(e_cov) and cost <= e_entry.trip_cost:
                if cov != e_cov or cost < e_entry.trip_cost:
                    dominated_keys.append((e_end, e_cov))
                    
        for dominated in dominated_keys:
            del cache[dominated]
            self.total_pruned += 1
        
        if new_key in cache:
            entry = cache[new_key]
            
            # Only count as new period if not merged multiple times in same period
            if entry.last_merged_period != current_period:
                entry.periods_seen += 1
                entry.periods_accepted += 1
                
            entry.hist_avg_reward = entry.hist_avg_reward + (reward - entry.hist_avg_reward) / max(1, entry.periods_seen)
            entry.hist_best_reward = max(entry.hist_best_reward, reward)
            entry.last_merged_period = current_period
            entry.last_accepted_period = current_period
            
            # Replace route if strictly better (covered by dominance checks, but catch exact matches that are cheaper)
            if cost < entry.trip_cost:
                entry.route = route
                entry.trip_cost = cost
        else:
            # Promote new entry
            entry = MemoryEntry(
                start_depot=start_depot,
                end_depot=end_depot,
                route=route,
                covered=cov,
                trip_cost=cost,
                current_reward=reward,
                hist_avg_reward=reward,
                hist_best_reward=reward,
                periods_seen=1,
                periods_accepted=1,
                last_merged_period=current_period,
                last_accepted_period=current_period
            )
            cache[new_key] = entry
            self.total_promoted += 1

    def prune(self, current_period: int):
        """
        Enforce max memory size by explicit redundancy checks and weak pruning.
        """
        for depot, cache in self.entries.items():
            keys_to_delete = []
            
            # 1. Stale entries
            for key, entry in cache.items():
                if current_period - entry.last_accepted_period >= 5:
                    keys_to_delete.append(key)
                    
            for key in keys_to_delete:
                del cache[key]
                self.total_pruned += 1
                
            # 2. Rank remaining by utility
            ranked = sorted(cache.values(), key=lambda x: x.utility, reverse=True)
            
            # 3. Overlap reduction: keep only the most robust structurally distinct options
            kept = []
            for entry in ranked:
                is_redundant = False
                for k in kept:
                    if k.end_depot == entry.end_depot and k.covered.issuperset(entry.covered):
                        is_redundant = True
                        break
                if not is_redundant:
                    kept.append(entry)
                else:
                    del cache[(entry.end_depot, entry.covered)]
                    self.total_pruned += 1
                    
            # 4. Enforce max top_k hard memory size
            if len(kept) > self.max_per_depot:
                for entry in kept[self.max_per_depot:]:
                    del cache[(entry.end_depot, entry.covered)]
                    self.total_pruned += 1


class DummyTripCache:
    """Non-functional trip cache for the baseline runs."""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.hits_accepted = 0
        self.hits_rejected = 0
        
    def get_trip(self, current_depot, remaining_edges, max_cost):
        self.misses += 1
        return None, None, None, 0.0
        
    def add_trip(self, start_depot, route, current_length, covered, all_depots, prob_reward):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §10  Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════
from magnetic_sa_mdrpp_rrv_cache_v2 import TripCache

def run_experiment(graph_dir: str, 
                   n_graphs: int,
                   n_periods: int,
                   n_repeats: int, 
                   sa_kwargs: dict):
                   
    meta_all = np.load(os.path.join(graph_dir, 'graph_params.npy'), allow_pickle=True)
    modes = ['exact_cache', 'intelligent_memory', 'baseline']
    all_results = []
    
    for graph_idx in range(n_graphs):
        meta = meta_all[graph_idx]
        graph_id = int(meta['graph_id'])
            
        capacity       = float(meta['vehicle_capacity'])
        recharge_time  = float(meta['recharge_time'])
        T_interval     = float(meta['time_interval']) / 2.0
        depot_nodes    = list(meta['depot_nodes'])
        numVehicle     = len(depot_nodes)
        
        pkl_path = os.path.join(graph_dir, f'{graph_id}.pickle')
        with open(pkl_path, 'rb') as fh:
            G = pickle.load(fh)
            
        max_flights = max(1, int(T_interval // max(recharge_time + capacity, 1e-9)))
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        
        # Load multi-period probabilities
        prob_npy_path = os.path.join(graph_dir, f'{graph_id}_edge_probs.npy')
        edge_probs_all = np.load(prob_npy_path, allow_pickle=True).item()
        
        log.info(f"\n============================================================")
        log.info(f"Graph {graph_id:>3} | nodes={G.number_of_nodes()} | edges={G.number_of_edges()} | depots={depot_nodes}")
        log.info(f"B={capacity:.0f} | TR={recharge_time:.0f} | T={T_interval:.0f} | max_flights={max_flights}")

        for mode in modes:
            log.info(f"--- Running Mode: {mode.upper()} ---")
            
            for repeat in range(n_repeats):
                log.info(f"  Graph {graph_id} | Mode {mode} | Repeat {repeat+1}/{n_repeats}")
                
                # Long-term memory persists *across periods* but resets *per repeat*
                LTM_MAX_SIZE = 100
                CMD_LTM_TOP_K = 20
                CMD_CACHE_SIZE = 500
                
                ltm = LongTermMemoryBank(max_per_depot=LTM_MAX_SIZE) if mode == 'intelligent_memory' else None
                
                for period in range(n_periods):
                    t0 = time.time()
                    
                    # build edgeProbs for current period
                    edge_probs = {}
                    for u, v, d in G.edges(data=True):
                        canon = tuple(sorted((u, v)))
                        edge_probs[canon] = edge_probs_all[canon][period]
                    
                    # Baseline uses no cache function
                    if mode == 'baseline':
                        trip_cache = DummyTripCache() # Keep dummy for log metrics compatibility
                    else:
                        trip_cache = TripCache(max_size=CMD_CACHE_SIZE)
                    
                    seeded_count = 0
                    if ltm is not None and period > 0:
                        selected = ltm.rescore_and_select(period, edge_probs, top_k=CMD_LTM_TOP_K)
                        seeded_count = ltm.seed_trip_cache(trip_cache, selected, set(depot_nodes))
                    
                    if mode == 'baseline':
                        best_trips, best_obj, best_prob, best_beta, _ = simulated_annealing_magnetic_no_cache(
                            G, dist, depot_nodes, edge_probs,
                            capacity, recharge_time, T_interval, max_flights,
                            trip_cache={},
                            **sa_kwargs
                        )
                    else:
                        best_trips, best_obj, best_prob, best_beta = simulated_annealing_magnetic(
                            G, dist, depot_nodes, edge_probs,
                            capacity, recharge_time, T_interval, max_flights,
                            trip_cache=trip_cache,
                            **sa_kwargs
                        )
                    
                    runtime = time.time() - t0
                    
                    if ltm is not None:
                        ltm.merge_trips(period, best_trips, edge_probs, G)
                        ltm.prune(period)
                        
                    # Calculate coverage
                    total_prob = sum(edge_probs.values())
                    covered_edges = set()
                    for vehicle_routes in best_trips:
                        for route in vehicle_routes:
                            for i in range(len(route) - 1):
                                u, v = route[i], route[i+1]
                                if u != v:
                                    canon = tuple(sorted((u, v)))
                                    if canon in edge_probs:
                                        covered_edges.add(canon)

                    covered_prob = sum(edge_probs[e] for e in covered_edges)
                    pct_covered = 100.0 * covered_prob / total_prob if total_prob > 0 else 0.0
                    unique_edges_covered = len(covered_edges)

                    # Collect metrics
                    hits = trip_cache.hits
                    misses = trip_cache.misses
                    accepts = getattr(trip_cache, 'hits_accepted', 0)
                    rejects = getattr(trip_cache, 'hits_rejected', 0)
                    
                    ltm_entries = sum(len(c) for c in ltm.entries.values()) if ltm else 0
                    promoted = ltm.total_promoted if ltm else 0
                    pruned = ltm.total_pruned if ltm else 0

                    rec = {
                        'graph_id': graph_id,
                        'mode': mode,
                        'repeat': repeat,
                        'period': period,
                        'obj': round(best_obj, 6),
                        'prob': round(best_prob, 6),
                        'beta': round(best_beta, 3),
                        'time_s': round(runtime, 3),
                        'unique_edges_covered': unique_edges_covered,
                        '%_covered': round(pct_covered, 2),
                        'cooling_rate': sa_kwargs.get('cooling_rate', np.nan),
                        'max_iterations': sa_kwargs.get('max_iterations', np.nan),
                        'no_improve_limit': sa_kwargs.get('no_improve_limit', np.nan),
                        'max_cache_size': CMD_CACHE_SIZE if mode != 'baseline' else 0,
                        'ltm_top_k': CMD_LTM_TOP_K if mode == 'intelligent_memory' else 0,
                        'ltm_max_size': LTM_MAX_SIZE if mode == 'intelligent_memory' else 0,
                        'cache_hits': hits,
                        'cache_misses': misses,
                        'cache_accepts': accepts,
                        'cache_rejects': rejects,
                        'ltm_entries': ltm_entries,
                        'ltm_seeded': seeded_count,
                        'ltm_promoted': promoted,
                        'ltm_pruned': pruned
                    }
                    all_results.append(rec)
                    
                    log.info(f"    Period {period} | obj: {best_obj:.4f} | time: {runtime:.2f}s | "
                             f"hits: {hits} ({accepts} acc) | miss: {misses} | "
                             f"seeded: {seeded_count} | LTM size: {ltm_entries}")

    return pd.DataFrame(all_results)


# ══════════════════════════════════════════════════════════════════════════════
# §14  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_dir  = os.path.join(script_dir, 'graphs_small_test')
    out_dir    = os.path.join(script_dir, 'memory_results_small')
    os.makedirs(out_dir, exist_ok=True)

    NUM_GRAPHS = 15
    NUM_PERIODS = 4

    log.info('Starting PRTMB experiment with %d graphs × %d periods × %d repeats',
             NUM_GRAPHS, NUM_PERIODS, N_REPEATS)

    results_df = run_experiment(
        graph_dir=graph_dir,
        n_graphs=NUM_GRAPHS,
        n_periods=NUM_PERIODS,
        n_repeats=N_REPEATS,
        sa_kwargs=SA_KWARGS
    )

    out_file_all = os.path.join(out_dir, 'all_results.csv')
    results_df.to_csv(out_file_all, index=False)
    log.info(f"All runs saved to {out_file_all}")

    # Compute summary statistics
    # Summary by graph_id, mode, period
    summary_cols = ['obj', 'time_s', 'unique_edges_covered', '%_covered']
    summary_df = results_df.groupby(['graph_id', 'mode', 'period'])[summary_cols].agg(['mean', 'std']).reset_index()
    # Flatten multi-level columns
    summary_df.columns = ['_'.join(c).strip('_') for c in summary_df.columns.values]

    out_file_summary = os.path.join(out_dir, 'summary_results.csv')
    summary_df.to_csv(out_file_summary, index=False)
    log.info(f"Summary saved to {out_file_summary}")