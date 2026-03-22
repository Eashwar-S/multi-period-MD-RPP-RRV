import sys
import re

file_path = r'c:\Users\eashw\Desktop\PhD_resesarch\multi-period-MD-RPP-RRV\multi_period_routing\run_gnn_routing_experiment_parallel.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add _run_mode_repeat before run_single_area
run_mode_func = """
# ══════════════════════════════════════════════════════════════════════════════
# Top-level Task Worker
# ══════════════════════════════════════════════════════════════════════════════

def _run_mode_repeat(args):
    (area_name, mode, repeat, n_periods,
     G, dist, depot_nodes, capacity, recharge_time,
     T_interval, max_flights, edge_probs_all, sa_kwargs) = args

    log.info(f"--- Mode: {mode.upper()} | Area: {area_name} | Repeat {repeat+1} ---")
    
    LTM_MAX_SIZE  = 100
    LTM_TOP_K     = 20
    CACHE_SIZE    = 500

    ltm = LongTermMemoryBank(max_per_depot=LTM_MAX_SIZE) if mode == 'intelligent_memory' else None
    
    results = []
    for period in range(n_periods):
        t0 = time.time()
        
        edge_probs = {}
        for edge_key, prob_list in edge_probs_all.items():
            p = float(prob_list[period]) if period < len(prob_list) else 0.0
            if isinstance(edge_key, str):
                try: edge_key = eval(edge_key)
                except Exception: continue
            canon = (min(edge_key[0], edge_key[1]), max(edge_key[0], edge_key[1]))
            edge_probs[canon] = p

        if mode == 'baseline':
            trip_cache = DummyTripCache()
        else:
            trip_cache = TripCache(max_size=CACHE_SIZE)

        seeded_count = 0
        if ltm is not None and period > 0:
            selected = ltm.rescore_and_select(period, edge_probs, top_k=LTM_TOP_K)
            seeded_count = ltm.seed_trip_cache(trip_cache, selected, set(depot_nodes), G)

        if mode == 'baseline':
            best_trips, best_obj, best_prob, best_beta, _ = simulated_annealing_magnetic_no_cache(
                G, dist, depot_nodes, edge_probs,
                capacity, recharge_time, T_interval, max_flights,
                trip_cache={}, **sa_kwargs
            )
        else:
            best_trips, best_obj, best_prob, best_beta = simulated_annealing_magnetic(
                G, dist, depot_nodes, edge_probs,
                capacity, recharge_time, T_interval, max_flights,
                trip_cache=trip_cache, **sa_kwargs
            )

        runtime = time.time() - t0

        if ltm is not None:
            ltm.merge_trips(period, best_trips, edge_probs, G)
            ltm.prune(period)

        total_prob = sum(edge_probs.values())
        covered_edges = set()
        for vehicle_routes in best_trips:
            for route in vehicle_routes:
                for i in range(len(route) - 1):
                    u, v = route[i], route[i+1]
                    if u != v:
                        canon = (min(u, v), max(u, v))
                        if canon in edge_probs:
                            covered_edges.add(canon)

        covered_prob = sum(edge_probs[e] for e in covered_edges)
        pct_covered  = 100.0 * covered_prob / total_prob if total_prob > 0 else 0.0

        hits     = trip_cache.hits
        misses   = trip_cache.misses
        accepts  = getattr(trip_cache, 'hits_accepted', 0)
        rejects  = getattr(trip_cache, 'hits_rejected', 0)
        ltm_entries = sum(len(c) for c in ltm.entries.values()) if ltm else 0
        promoted    = ltm.total_promoted if ltm else 0
        pruned      = ltm.total_pruned if ltm else 0

        rec = {
            'area': area_name,
            'mode': mode,
            'repeat': repeat,
            'period': period,
            'obj': round(best_obj, 6),
            'prob': round(best_prob, 6),
            'beta': round(best_beta, 3),
            'time_s': round(runtime, 3),
            'unique_edges_covered': len(covered_edges),
            '%_covered': round(pct_covered, 2),
            'capacity_s': capacity,
            'T_interval_s': T_interval,
            'n_depots': len(depot_nodes),
            'max_flights': max_flights,
            'cache_hits': hits,
            'cache_misses': misses,
            'cache_accepts': accepts,
            'cache_rejects': rejects,
            'ltm_entries': ltm_entries,
            'ltm_seeded': seeded_count,
            'ltm_promoted': promoted,
            'ltm_pruned': pruned,
        }
        results.append(rec)

        log.info(f"[{mode.upper():20s} | rep {repeat+1} | p{period:2d}] obj: {best_obj:.4f} | "
                 f"cov: {pct_covered:.1f}% | time: {runtime:.2f}s | "
                 f"hits: {hits} | seeded: {seeded_count} | LTM: {ltm_entries}")

    return results

def run_single_area"""

content = content.replace("def run_single_area", run_mode_func)

# 2. Modify run_experiment
run_exp_target = """def run_experiment(instances_dir: str, n_repeats: int, sa_kwargs: dict,
                   area_filter: str = None):"""
run_exp_replace = """def run_experiment(instances_dir: str, n_repeats: int, sa_kwargs: dict,
                   area_filter: str = None, n_workers: int = 1):"""
content = content.replace(run_exp_target, run_exp_replace)

modes_loop_target = r"# ── Run modes ──────────────────────────────────────────────────────────.*?(?=    return all_results)"
modes_loop_replace = """# ── Run modes ──────────────────────────────────────────────────────────
        task_args = []
        for mode in modes:
            for repeat in range(n_repeats):
                task_args.append((
                    area_name, mode, repeat, n_periods,
                    G, dist, depot_nodes, capacity, recharge_time,
                    T_interval, max_flights, edge_probs_all, sa_kwargs
                ))

        log.info(f"Dispatching {len(task_args)} tasks for {area_slug} "
                 f"(workers={n_workers})...")
        
        area_results = []
        if n_workers > 1:
            with mp.get_context('spawn').Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(_run_mode_repeat, task_args):
                    area_results.extend(res)
        else:
            for t_args in task_args:
                area_results.extend(_run_mode_repeat(t_args))
                
        all_results.extend(area_results)

"""
content = re.sub(modes_loop_target, modes_loop_replace, content, flags=re.DOTALL)


# 3. Modify single-area call in __main__
main_target = """    # Single-area mode stays simple
    if args.area:
        log.info(f"Single-area mode: {args.area}")
        results = run_experiment(
            instances_dir=instances_dir,
            n_repeats=N_REPEATS,
            sa_kwargs=SA_KWARGS,
            area_filter=args.area,
        )"""

main_replace = """    # Single-area mode stays simple
    if args.area:
        log.info(f"Single-area mode: {args.area}")
        results = run_experiment(
            instances_dir=instances_dir,
            n_repeats=N_REPEATS,
            sa_kwargs=SA_KWARGS,
            area_filter=args.area,
            n_workers=args.jobs,
        )"""
content = content.replace(main_target, main_replace)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch applied")
