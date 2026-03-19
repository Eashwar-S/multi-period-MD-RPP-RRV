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
from magnetic_sa_mdrpp_rrv_cache import (
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
    max_iterations=1000,
    no_improve_limit=200,
)

# Experiment configuration
NUM_INSTANCES   = 4
N_REPEATS       = 10   # independent SA repeats per (mode × instance) for mean±std
RANDOM_SEED_BASE = 42
PROB_PERTURBATION_SCALE = 0.25   # Gaussian σ for probability perturbation
PROB_MIN_CLIP = 0.05
PROB_MAX_CLIP = 0.95
GRAPH_ID_TO_RUN = 15              # which graph to use for the 7-instance experiment


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
    periods_selected: int = 0
    last_period_used: int = 0

    @property
    def utility(self) -> float:
        """
        Ranking score for long-term memory entries.
        Utility is based on:
        - current-period reward density (heavy weight)
        - historical average reward density
        - acceptance / selection rate
        """
        if self.trip_cost <= 1e-9:
            return 0.0
            
        cur_density = self.current_reward / self.trip_cost
        hist_density = self.hist_avg_reward / self.trip_cost
        
        # acceptance rate (if seen more than once)
        acc_rate = self.periods_selected / max(1, self.periods_seen)
        
        # Weighted utility
        return (cur_density * 0.5) + (hist_density * 0.3) + (acc_rate * 0.2)


class LongTermMemoryBank:
    """
    Cross-period structural memory bank.
    Stores robust depot-to-depot trips evaluated and promoted over multiple periods.
    """
    def __init__(self, max_per_depot=100):
        self.max_per_depot = max_per_depot
        
        # start_depot -> { covered_edges : MemoryEntry }
        self.entries: Dict[int, Dict[frozenset, MemoryEntry]] = {}
        
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
            for cov, entry in cache.items():
                reward = sum(current_probs.get(e, 0.0) for e in cov)
                entry.current_reward = reward
                
            # Sort by utility descending
            ranked = sorted(cache.values(), key=lambda x: x.utility, reverse=True)
            top_entries = ranked[:top_k]
            
            for e in top_entries:
                e.periods_selected += 1
                e.last_period_used = current_period
                
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
        
        if cov in cache:
            # Update existing entry
            entry = cache[cov]
            
            # Only count as new period if not merged multiple times in same period
            if entry.last_period_used != current_period:
                entry.periods_seen += 1
                
            entry.hist_avg_reward = entry.hist_avg_reward + (reward - entry.hist_avg_reward) / max(1, entry.periods_seen)
            entry.hist_best_reward = max(entry.hist_best_reward, reward)
            entry.last_period_used = current_period
            
            # Replace route if strictly better (cheaper for same coverage)
            if cost < entry.trip_cost:
                entry.route = route
                entry.trip_cost = cost
                entry.end_depot = end_depot
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
                last_period_used=current_period
            )
            cache[cov] = entry
            self.total_promoted += 1

    def prune(self):
        """
        Enforce max memory size by pruning the weakest entries per depot.
        """
        for depot, cache in self.entries.items():
            if len(cache) > self.max_per_depot:
                # Rank to keep top entries
                ranked = sorted(cache.values(), key=lambda x: x.utility, reverse=True)
                
                # Prune remainder
                for entry in ranked[self.max_per_depot:]:
                    del cache[entry.covered]
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
from magnetic_sa_mdrpp_rrv_cache import TripCache

def run_experiment(graph_id: int, 
                   n_instances: int, 
                   n_repeats: int, 
                   sa_kwargs: dict,
                   graph_dir: str):
                   
    meta_all = np.load(os.path.join(graph_dir, 'graph_params.npy'), allow_pickle=True)
    
    meta = None
    for m in meta_all:
        if int(m['graph_id']) == graph_id:
            meta = m
            break
            
    if meta is None:
        raise ValueError(f"Graph {graph_id} not found in meta")
        
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
    
    log.info(f"Graph {graph_id:>3} | nodes={G.number_of_nodes()} | edges={G.number_of_edges()} | depots={depot_nodes}")
    log.info(f"B={capacity:.0f} | TR={recharge_time:.0f} | T={T_interval:.0f} | max_flights={max_flights}")

    # Generate edge probability instances
    prob_instances = generate_probability_instances(G, n_instances, RANDOM_SEED_BASE)
    
    modes = ['exact_cache', 'intelligent_memory', 'baseline']
    all_results = []
    
    for mode in modes:
        log.info(f"--- Running Mode: {mode.upper()} ---")
        
        for repeat in range(n_repeats):
            log.info(f"  Repeat {repeat+1}/{n_repeats}")
            
            # Long-term memory persists *across instances* but resets *per repeat*
            ltm = LongTermMemoryBank(max_per_depot=1000) if mode == 'intelligent_memory' else None
            
            for inst_idx, edge_probs in enumerate(prob_instances):
                t0 = time.time()
                
                # Baseline uses no cache function
                if mode == 'baseline':
                    trip_cache = DummyTripCache() # Keep dummy for log metrics compatibility
                    
                else:
                    trip_cache = TripCache(max_size=500)
                
                seeded_count = 0
                if ltm is not None and inst_idx > 0:
                    selected = ltm.rescore_and_select(inst_idx, edge_probs, top_k=100)
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
                    ltm.merge_trips(inst_idx, best_trips, edge_probs, G)
                    ltm.prune()
                    
                # Collect metrics
                hits = trip_cache.hits
                misses = trip_cache.misses
                accepts = getattr(trip_cache, 'hits_accepted', 0)
                rejects = getattr(trip_cache, 'hits_rejected', 0)
                
                ltm_entries = sum(len(c) for c in ltm.entries.values()) if ltm else 0
                promoted = ltm.total_promoted if ltm else 0
                pruned = ltm.total_pruned if ltm else 0

                rec = {
                    'mode': mode,
                    'repeat': repeat,
                    'instance': inst_idx,
                    'obj': round(best_obj, 6),
                    'prob': round(best_prob, 6),
                    'beta': round(best_beta, 3),
                    'time_s': round(runtime, 3),
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
                
                log.info(f"    Inst {inst_idx} | obj: {best_obj:.4f} | time: {runtime:.2f}s | "
                         f"hits: {hits} ({accepts} acc) | miss: {misses} | "
                         f"seeded: {seeded_count} | LTM size: {ltm_entries}")

    return pd.DataFrame(all_results)


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

    results_df = run_experiment(
        graph_id=GRAPH_ID_TO_RUN,
        n_instances=NUM_INSTANCES,
        n_repeats=N_REPEATS,
        sa_kwargs=SA_KWARGS,
        graph_dir=graph_dir
    )

    # Compute summary statistics
    summary_df = results_df.groupby(['instance', 'mode'])[['obj', 'time_s']].agg(['mean', 'std']).reset_index()
    # Flatten multi-level columns
    summary_df.columns = ['_'.join(c).strip('_') for c in summary_df.columns.values]

    out_file = os.path.join(out_dir, f'intelligent_memory_results_graph_{GRAPH_ID_TO_RUN}.xlsx')
    
    with pd.ExcelWriter(out_file) as writer:
        results_df.to_excel(writer, sheet_name='All Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    log.info(f"Experiment complete. Results saved to {out_file}")

    # Generate error plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    instances = summary_df['instance'].unique()
    modes = summary_df['mode'].unique()
    
    x = np.arange(len(instances))
    width = 0.35
    
    for i, mode in enumerate(modes):
        mode_data = summary_df[summary_df['mode'] == mode]
        
        # Align bars side by side
        offset = (i - 0.5) * width if len(modes) == 3 else 0
        
        plot_x = x[np.isin(instances, mode_data['instance'])]
        
        # Objective Plot
        axes[0].bar(plot_x + offset, mode_data['obj_mean'], width, 
                    yerr=mode_data['obj_std'], label=mode, capsize=5, alpha=0.8)
        
        # Time Plot
        axes[1].bar(plot_x + offset, mode_data['time_s_mean'], width, 
                    yerr=mode_data['time_s_std'], label=mode, capsize=5, alpha=0.8)
                    
    # Format Objective Plot
    axes[0].set_title('Objective Value per Instance (Mean ± Std)')
    axes[0].set_xlabel('Instance ID')
    axes[0].set_ylabel('Objective Value (Higher is Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(instances)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format Time Plot
    axes[1].set_title('Runtime per Instance (Mean ± Std)')
    axes[1].set_xlabel('Instance ID')
    axes[1].set_ylabel('Time (sec)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(instances)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f'intelligent_memory_plot_graph_{GRAPH_ID_TO_RUN}.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    log.info(f"Plots saved to {plot_file}")