#!/usr/bin/env python3
"""
Multi-Trip Router (Parameter-Free), with:
  - Cycle-proof single-trip growth (frontier-progress + no-backtrack)
  - Coverage-aware finisher (relocation-with-coverage; orienteering-style on meta-graph)
  - Start-depot retargeting INSIDE the same trip (no standalone relocations)
  - Look-ahead scoring of certified end depots (prefer cheapest next cycle)
  - One-step backtrack if retargeting fails
  - Positioning-trip fallback (depot->depot within capacity) to unlock coverage
  - **NEW**: Per-state depot banlist for zero-coverage attempts

Goal: minimize sum of trip lengths, exploring new depots when a depot yields zero coverage for the same state.
"""

import re
import math
import networkx as nx
from math import exp
from typing import List, Tuple, Set, Dict, Optional

# ---------- Parsing ----------

def parse_txt_file(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def extract_int(key):
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+)", ln)
                if m: return int(m[0])
        return None

    def extract_float(key):
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+(?:\.\d+)?)", ln)
                if m: return float(m[0])
        return None

    def extract_list_depots():
        for ln in lines:
            if ln.upper().startswith("DEPOT"):
                nums = re.findall(r"(\d+)", ln)
                return [int(x) for x in nums]
        return []

    nV = extract_int("NUMBER OF VERTICES")
    capacity = extract_float("VEHICLE CAPACITY") or 0.0
    depots = extract_list_depots()

    req_section = False
    nonreq_section = False
    required_edges = []
    weights = {}

    for ln in lines:
        if ln.upper().startswith("LIST_REQUIRED_EDGES"):
            req_section, nonreq_section = True, False
            continue
        if ln.upper().startswith("LIST_NON_REQUIRED_EDGES"):
            req_section, nonreq_section = False, True
            continue
        if ln.upper().startswith("FAILURE_SCENARIO"):
            req_section = nonreq_section = False
            continue

        m = re.match(r"\((\d+),\s*(\d+)\)\s*edge\s*weight\s*([0-9.]+)", ln, re.IGNORECASE)
        if m:
            u, v, w = int(m.group(1)), int(m.group(2)), float(m.group(3))
            wkey = tuple(sorted((u, v)))
            weights[wkey] = w
            if req_section:
                required_edges.append((u, v))

    G = nx.Graph()
    if nV:
        G.add_nodes_from(range(1, nV + 1))
    for (a, b), w in weights.items():
        G.add_edge(a, b, weight=w)

    return G, required_edges, depots, capacity


# ---------- Utility: Look-ahead & Feasibility oracles ----------

def best_next_cycle_cost_for_depot(
    G: nx.Graph,
    dist: Dict[int, Dict[int, float]],
    depots: Set[int],
    cap: float,
    depot: int,
    remaining: Set[Tuple[int, int]],
) -> float:
    if not remaining:
        return 0.0
    best = float('inf')
    row = dist.get(depot, {})
    for (a, b) in remaining:
        w = G[a][b]['weight']
        da = row.get(a, float('inf'))
        db = row.get(b, float('inf'))
        tail_b = min((dist.get(b, {}).get(d2, float('inf')) for d2 in depots), default=float('inf'))
        tail_a = min((dist.get(a, {}).get(d2, float('inf')) for d2 in depots), default=float('inf'))
        cand1 = da + w + tail_b
        cand2 = db + w + tail_a
        best = min(best, cand1, cand2)
    return best


def exists_feasible_next_from_depot(
    G: nx.Graph,
    dist: Dict[int, Dict[int, float]],
    depots: Set[int],
    cap: float,
    depot: int,
    remaining: Set[Tuple[int, int]],
) -> bool:
    return best_next_cycle_cost_for_depot(G, dist, depots, cap, depot, remaining) <= cap


# ---------- Router ----------

class MagneticFieldRouter:
    def __init__(self, graph: nx.Graph, dist: Dict[int, Dict[int, float]], start_depot: int, depots: List[int], capacity: float,
                 alpha: float = 1.0, gamma: float = 1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.depots = set(depots)
        self.capacity = float(capacity)
        self.alpha = alpha
        self.gamma = gamma
        self._dist = dist

    def _min_dist_to_any_depot(self, node: int) -> float:
        INF = float('inf')
        row = self._dist.get(node, {})
        return min((row.get(d, INF) for d in self.depots), default=INF)

    def _dist_to_frontier(self, node: int, req_set_remaining: Set[Tuple[int, int]]) -> float:
        INF = float('inf')
        best = INF
        row = self._dist.get(node, {})
        for (a, b) in req_set_remaining:
            da = row.get(a, INF); db = row.get(b, INF)
            if da < best: best = da
            if db < best: best = db
            if best == 0: break
        return best

    def calculate_edge_score(self, edge: Tuple[int, int],
                             req_remaining: List[Tuple[int, int]],
                             current_length: float,
                             is_new_required: bool):
        u, v = edge
        w_uv = self.graph[u][v]['weight']
        w_cap = min(max(current_length / max(self.capacity, 1e-9), 0.0), 1.0)
        if is_new_required:
            P = 1.0
        else:
            req_set = {tuple(sorted(e)) for e in req_remaining}
            Fv = self._dist_to_frontier(v, req_set)
            P = 0.0 if math.isinf(Fv) else exp(-1.0 * Fv)
        d_v = self._min_dist_to_any_depot(v)
        D = 0.0 if math.isinf(d_v) else exp(-1.0 * (d_v / max(self.capacity, 1e-9)))
        final = (1.0 - w_cap) * P + w_cap * D
        return {"edge_weight": w_uv, "final_score": final}

    def find_route_with_magnetic_scoring(self, required_edges, verbose: bool = False):
        current_route = [self.start_depot]
        current_length = 0.0
        required_covered = set()
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0

        # === NEW: anchors ===
        depot_anchor_idx = 0
        depot_anchor_node = self.start_depot
        lock_idx = 0  # freeze prefix up to here (after last covered required edge)

        req_set_all = {tuple(sorted(e)) for e in required_edges}
        depot_depot_covered_this_trip = set()

        def _path_cost(path):
            if len(path) < 2: return 0.0
            s = 0.0
            for i in range(1, len(path)):
                a, b = path[i-1], path[i]
                s += self.graph[a][b]['weight']
            return s

        while len(required_covered) < len(req_set_all) and iteration_count < max_iterations:
            iteration_count += 1
            u = current_route[-1]
            req_set_remaining = req_set_all - required_covered
            if not req_set_remaining:
                break

            Fu = self._dist_to_frontier(u, req_set_remaining)
            candidates = []

            for v in self.graph.neighbors(u):
                edge = (u, v)
                e_sorted = tuple(sorted(edge))
                w_uv = self.graph[u][v]['weight']
                depot_after = self._min_dist_to_any_depot(v)
                is_new_required = (e_sorted in req_set_remaining)

                # --- capacity feasibility, possibly with a refined prefix ---
                feas_prefix_len = current_length
                feas_refined_slice = None  # (start_idx, stop_idx, replacement_path)

                if is_new_required:
                    # splice only the unfrozen suffix; keep trip head (depot) and covered edges intact
                    splice_start_idx = max(depot_anchor_idx, lock_idx)
                    splice_head_node = current_route[splice_start_idx]  # either last depot or after last covered edge

                    # shortest path from splice head to current 'u'
                    sp_path = nx.shortest_path(self.graph, splice_head_node, u, weight='weight')
                    sp_len  = _path_cost(sp_path)

                    # current slice we would replace
                    curr_slice = current_route[splice_start_idx:]
                    curr_slice_len = _path_cost(curr_slice)

                    if sp_len + 1e-9 < curr_slice_len:
                        feas_prefix_len = current_length - curr_slice_len + sp_len
                        feas_refined_slice = (splice_start_idx, len(current_route), sp_path)

                # capacity check using possibly refined prefix
                if feas_prefix_len + w_uv + depot_after > self.capacity + 1e-9:
                    continue

                # block trivial backtrack if not covering new required
                if len(current_route) >= 2 and v == current_route[-2] and not is_new_required:
                    continue

                if (u in self.depots and v in self.depots and
                    e_sorted in depot_depot_covered_this_trip and not is_new_required):
                    continue

                Fv = self._dist_to_frontier(v, req_set_remaining)
                if not is_new_required and not (Fv < Fu):
                    continue

                sd = self.calculate_edge_score(edge, list(req_set_remaining), current_length, is_new_required)
                candidates.append({
                    'edge': edge, 'neighbor': v, 'is_new_required': is_new_required,
                    'score_data': sd, 'Fv': Fv,
                    'feas_prefix_len': feas_prefix_len,
                    'feas_refined_slice': feas_refined_slice
                })

            if not candidates:
                # keep your existing fallback block here (SP to nearest uncovered endpoint)
                best_path, best_len = None, float('inf')
                for (a, b) in req_set_remaining:
                    for t in (a, b):
                        try:
                            path = nx.shortest_path(self.graph, u, t, weight='weight')
                            plen = nx.shortest_path_length(self.graph, u, t, weight='weight')
                            depot_after = self._min_dist_to_any_depot(t)
                            if current_length + plen + depot_after <= self.capacity and plen < best_len:
                                best_path, best_len = path[1:], plen
                        except nx.NetworkXNoPath:
                            continue
                if best_path:
                    for nxt in best_path:
                        prev = current_route[-1]
                        e_sorted2 = tuple(sorted((prev, nxt)))
                        if prev in self.depots and nxt in self.depots and e_sorted2 in req_set_remaining:
                            depot_depot_covered_this_trip.add(e_sorted2)
                        current_route.append(nxt)
                        current_length += self.graph[prev][nxt]['weight']
                        # update depot anchor when we actually step onto a depot
                        if nxt in self.depots:
                            depot_anchor_idx = len(current_route) - 1
                            depot_anchor_node = nxt
                    continue
                break

            # choose
            candidates.sort(key=lambda c: (not c['is_new_required'], c['Fv'], -c['score_data']['final_score']))
            best = candidates[0]
            v = best['neighbor']

            # --- splice refinement in place (only for required edge) ---
            if best['is_new_required'] and best['feas_refined_slice'] is not None:
                a, b, sp_path = best['feas_refined_slice']
                old_slice = current_route[a:b]
                old_len   = _path_cost(old_slice)
                sp_len    = _path_cost(sp_path)
                current_route = current_route[:a] + sp_path
                current_length = current_length - old_len + sp_len
                u = current_route[-1]  # tail may have changed

            # commit u->v
            current_route.append(v)
            current_length += best['score_data']['edge_weight']

            # update anchors
            if v in self.depots:
                depot_anchor_idx = len(current_route) - 1
                depot_anchor_node = v

            if best['is_new_required']:
                e_sorted = tuple(sorted(best['edge']))
                required_covered.add(e_sorted)
                if best['edge'][0] in self.depots and best['edge'][1] in self.depots:
                    depot_depot_covered_this_trip.add(e_sorted)
                # freeze everything up through this new vertex to preserve covered-edge order
                lock_idx = len(current_route) - 1

            if verbose:
                print(f'current route - {current_route}, num req edges - {len(required_covered)}')

        # finishing (unchanged logic)
        u = current_route[-1]
        req_set_remaining = req_set_all - required_covered
        finish_route, finish_cost, newly_covered = self._finish_with_coverage_orienteering_certified(
            start_node=u, base_cost=current_length, req_set_remaining=set(req_set_remaining)
        )
        if finish_route and len(finish_route) > 1:
            for i in range(1, len(finish_route)):
                a, b = finish_route[i-1], finish_route[i]
                current_route.append(b)
            current_length = finish_cost
            required_covered |= newly_covered

        if current_route[-1] not in self.depots:
            d, dist_d = self._nearest_depot_within_budget(current_route[-1], self.capacity - current_length)
            if d is not None:
                current_route += list(self._shortest_path_nodes(current_route[-1], d)[1:])
                current_length += dist_d

        # print(f'current route - {current_route}')
        # Safety: head must be a depot
        assert current_route and current_route[0] in self.depots, "Trip head must be a depot."
        return current_route, current_length, len(required_covered)



    def _finish_with_coverage_orienteering_certified(self, start_node: int, base_cost: float,
                                                     req_set_remaining: Set[Tuple[int, int]]):
        budget = self.capacity - base_cost
        if budget <= 1e-12:
            if start_node in self.depots:
                return [start_node], base_cost, set()
            return [start_node], base_cost, set()

        req_list = sorted(list({tuple(sorted(e)) for e in req_set_remaining}))
        if not req_list:
            d, dlen = self._nearest_depot_within_budget(start_node, budget)
            if d is None:
                return [start_node], base_cost, set()
            path = self._shortest_path_nodes(start_node, d)
            return path, base_cost + dlen, set()

        idx_to_edge: Dict[int, Tuple[int, int, float]] = {}
        endpoint_to_edges: Dict[int, List[int]] = {}
        for i, (a, b) in enumerate(req_list):
            w = self.graph[a][b]['weight']
            idx_to_edge[i] = (a, b, w)
            endpoint_to_edges.setdefault(a, []).append(i)
            endpoint_to_edges.setdefault(b, []).append(i)

        meta_nodes = set()
        for (a, b) in req_list:
            meta_nodes.add(a); meta_nodes.add(b)
        meta_nodes |= set(self.depots)
        meta_nodes.add(start_node)

        def dist(x, y):
            return self._dist.get(x, {}).get(y, float('inf'))

        from collections import deque
        Label = Tuple[int, frozenset]
        best: Dict[Label, Tuple[float, float]] = {}
        parent: Dict[Label, Tuple[Optional[Label], Tuple[str, int]]] = {}

        start_label: Label = (start_node, frozenset())
        best[start_label] = (0.0, 0.0)
        parent[start_label] = (None, ("start", -1))
        q = deque([start_label])

        def dominates(a_cost, a_prize, b_cost, b_prize):
            return (a_cost <= b_cost and a_prize >= b_prize) and (a_cost < b_cost or a_prize > b_prize)

        while q:
            node, covered = q.popleft()
            cost_used, prize = best[(node, covered)]
            remaining = budget - cost_used
            if remaining < -1e-12:
                continue

            if node in endpoint_to_edges:
                for e_idx in endpoint_to_edges[node]:
                    if e_idx in covered:
                        continue
                    a, b, w_e = idx_to_edge[e_idx]
                    nxt = b if node == a else a if node == b else None
                    if nxt is None:
                        continue
                    new_cost = cost_used + w_e
                    if new_cost <= budget + 1e-12:
                        new_cov = frozenset(set(covered) | {e_idx})
                        new_prize = prize + w_e
                        lbl = (nxt, new_cov)
                        old = best.get(lbl)
                        if old is None or dominates(new_cost, new_prize, old[0], old[1]):
                            best[lbl] = (new_cost, new_prize)
                            parent[lbl] = ((node, covered), ("cover", e_idx))
                            q.append(lbl)

            for t in meta_nodes:
                if t == node:
                    continue
                d_nt = dist(node, t)
                if not math.isfinite(d_nt):
                    continue
                new_cost = cost_used + d_nt
                if new_cost <= budget + 1e-12:
                    lbl = (t, covered)
                    old = best.get(lbl)
                    if old is None or dominates(new_cost, prize, old[0], old[1]):
                        best[lbl] = (new_cost, prize)
                        parent[lbl] = ((node, covered), ("move", t))
                        q.append(lbl)

        best_end = None
        best_score = None  # (-prize, tail_len, total_cost, next_cycle_cost)
        best_end_depot = None

        for (node, covered_idxs), (cost_used, prize) in best.items():
            newly_cov_edges = {tuple(sorted((idx_to_edge[i][0], idx_to_edge[i][1]))) for i in covered_idxs}
            rem_after_tail = set(req_set_remaining) - newly_cov_edges

            def next_cycle_cost(d: int) -> float:
                return best_next_cycle_cost_for_depot(
                    self.graph, self._dist, self.depots, self.capacity, d, rem_after_tail
                )

            if node in self.depots:
                ncc = next_cycle_cost(node)
                if ncc > self.capacity:
                    continue
                tail = 0.0
                total_cost = base_cost + cost_used
                score = (-prize, tail, total_cost, ncc)
                if best_score is None or score < best_score:
                    best_score = score
                    best_end = (node, covered_idxs)
                    best_end_depot = node
            else:
                for d in self.depots:
                    dlen = dist(node, d)
                    if not math.isfinite(dlen):
                        continue
                    if cost_used + dlen <= budget:
                        ncc = next_cycle_cost(d)
                        if ncc > self.capacity:
                            continue
                        total_cost = base_cost + cost_used + dlen
                        score = (-prize, dlen, total_cost, ncc)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_end = (node, covered_idxs)
                            best_end_depot = d

        if best_end is None:
            return [start_node], base_cost, set()

        route_nodes = [best_end[0]]
        cur = best_end
        while parent[cur][0] is not None:
            prev, _ = parent[cur]
            route_nodes.append(prev[0])
            cur = prev
        route_nodes.reverse()

        real_path = [start_node]
        for i in range(1, len(route_nodes)):
            a, b = route_nodes[i - 1], route_nodes[i]
            seg = self._shortest_path_nodes(a, b)
            real_path.extend(seg[1:])
        end_node = route_nodes[-1]
        if end_node != best_end_depot:
            seg = self._shortest_path_nodes(end_node, best_end_depot)
            real_path.extend(seg[1:])

        new_cost = base_cost
        newly_covered = set()
        for i in range(1, len(real_path)):
            a, b = real_path[i - 1], real_path[i]
            w = self.graph[a][b]['weight']
            new_cost += w
            e_sorted = tuple(sorted((a, b)))
            if e_sorted in req_set_remaining:
                newly_covered.add(e_sorted)

        return real_path, new_cost, newly_covered

    def _shortest_path_nodes(self, a: int, b: int) -> List[int]:
        if a == b: return [a]
        return nx.shortest_path(self.graph, a, b, weight='weight')

    def _nearest_depot_within_budget(self, u: int, budget: float):
        best_d, best_len = None, float('inf')
        row = self._dist.get(u, {})
        for d in self.depots:
            L = row.get(d, float('inf'))
            if math.isfinite(L) and L <= budget and L < best_len:
                best_d, best_len = d, L
        return best_d, best_len


# ---------- Start-depot retargeting (skip banned) ----------

def plan_with_start_retargeting(
    G: nx.Graph, dist, depots: List[int], capacity: float,
    current_start: int, req_set_remaining: Set[Tuple[int, int]],
    banned: Set[int]
):
    """
    Try all depots s* not in banned:
      deadhead current_start->s* + cover from s* (reduced capacity).
    Choose minimal total cost with coverage>0.
    """
    dist_row = dict(nx.single_source_dijkstra_path_length(G, current_start, weight='weight'))
    best = None

    for s_star in depots:
        if s_star in banned:  # skip banned depots for this state
            continue
        D = 0.0 if s_star == current_start else dist_row.get(s_star, float('inf'))
        if not math.isfinite(D) or D >= capacity - 1e-12:
            continue

        router2 = MagneticFieldRouter(G, dist=dist, start_depot=s_star, depots=depots,
                                      capacity=capacity - D, alpha=1.0, gamma=1.0)
        trip2_route, trip2_cost_reduced, _ = router2.find_route_with_magnetic_scoring(list(req_set_remaining), verbose=False)

        req_set = set(req_set_remaining)
        route_edge_seq = [tuple(sorted((trip2_route[i], trip2_route[i+1]))) for i in range(len(trip2_route)-1)]
        covered2 = {e for e in route_edge_seq if e in req_set}
        if not covered2:
            continue

        if s_star == current_start:
            full_route = trip2_route
        else:
            deadhead_path = nx.shortest_path(G, current_start, s_star, weight='weight')
            full_route = deadhead_path + trip2_route[1:]

        full_cost = D + trip2_cost_reduced
        cand = (full_cost, -len(covered2), s_star, full_route, covered2)
        if best is None or cand < best:
            best = cand

    if best is None:
        return None, None, None

    _, _, _, full_route, covered2 = best
    full_cost = best[0]
    return full_route, full_cost, covered2


# ---------- Positioning trip planner (skip banned) ----------

def plan_positioning_trip_to_certified_depot(
    G: nx.Graph,
    dist: Dict[int, Dict[int, float]],
    depots: List[int],
    capacity: float,
    current_start: int,
    req_set_remaining: Set[Tuple[int, int]],
    banned: Set[int],
):
    """
    Choose d not in banned, d!=current_start, with:
      - shortest_path_length(current_start,d) <= capacity
      - exists_feasible_next_from_depot(..., d, remaining) True
    Returns (route, cost) or (None, None).
    """
    best = None
    row = dist.get(current_start, {})
    for d in depots:
        if d == current_start or d in banned:
            continue
        move_cost = row.get(d, float('inf'))
        if not math.isfinite(move_cost) or move_cost > capacity:
            continue
        if not exists_feasible_next_from_depot(G, dist, set(depots), capacity, d, req_set_remaining):
            continue
        path = nx.shortest_path(G, current_start, d, weight='weight')
        cand = (move_cost, d, path)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None, None
    move_cost, d, path = best
    return path, move_cost


# ---------- Multi-trip loop (with per-state banlist) ----------

def run_intelligent_tuning_demo():

    for sce in range(21, 113):
        print(f'\nRunning scenario - {sce}')
        # scenario_txt_path = f"gdb_failure_scenarios/gdb.{sce}.txt"   # change to your file name if needed
        # scenario_txt_path = f"bccm_failure_scenarios/bccm.{sce}.txt"   # change to your file name if needed
        scenario_txt_path = f"eglese_failure_scenarios/eglese.{sce}.txt"   # change to your file name if needed
        G, req_edges, depots, cap = parse_txt_file(scenario_txt_path)
        # req_edges = [[70,71]]
        print(f'vehicle capacity - {cap}')
        print(f'required edges - {req_edges}')
        print(f'depot nodes - {depots}')
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        req_set_remaining = set(tuple(sorted(e)) for e in req_edges)
        START_DEPOT = depots[4]
        MAX_VEHICLE_CAPACITY = cap

        trips = []
        total_cost = 0.0
        trip_idx = 0
        MAX_TRIPS_GUARD = max(2 * len(req_set_remaining), 50)

        current_start = START_DEPOT
        _seen_backtracks: Set[Tuple[frozenset, int]] = set()
        _seen_repositions: Set[Tuple[frozenset, int]] = set()

        # NEW: per-state banlist { state: set(depots_that_yielded_zero_coverage_for_that_state) }
        banned_by_state: Dict[frozenset, Set[int]] = {}

        while req_set_remaining and trip_idx < MAX_TRIPS_GUARD:
            state_key_only = frozenset(req_set_remaining)
            banned = banned_by_state.setdefault(state_key_only, set())

            router = MagneticFieldRouter(
                graph=G,
                dist=dist,
                start_depot=current_start,
                depots=depots,
                capacity=MAX_VEHICLE_CAPACITY,
                alpha=1.0, gamma=1.0
            )

            route, trip_cost, _ = router.find_route_with_magnetic_scoring(list(req_set_remaining), verbose=False)

            route_edges = [tuple(sorted((route[i], route[i+1]))) for i in range(len(route)-1)]
            covered_now = {e for e in route_edges if e in req_set_remaining}
            # print(route, trip_cost, covered_now)

            if not covered_now:
                # mark this depot as "bad for this state"
                banned.add(current_start)

                # 1) Try retargeting inside the SAME trip, skipping banned depots
                ret_route, ret_cost, ret_covered = plan_with_start_retargeting(
                    G, dist, depots, MAX_VEHICLE_CAPACITY, current_start, req_set_remaining, banned
                )
                # print(ret_route, ret_cost, ret_covered)

                if ret_route is None:
                    # 2) One-step backtrack
                    print("[Halt] No coverage achievable within one capacity cycle from any depot. Attempting one-step backtrack...")

                    state_key = (state_key_only, current_start)
                    if state_key in _seen_backtracks:
                        # 3) Positioning trip fallback, skipping banned depots
                        print("[Stop] Backtrack led to the same state again; attempting positioning trip fallback...")
                        if state_key in _seen_repositions:
                            print("[Stop] Positioning fallback already tried for this state; declaring remainder infeasible.")
                            break
                        _seen_repositions.add(state_key)

                        pos_route, pos_cost = plan_positioning_trip_to_certified_depot(
                            G, dist, depots, MAX_VEHICLE_CAPACITY, current_start, req_set_remaining, banned
                        )
                        if pos_route is None:
                            print("[Stop] No valid positioning depot within one capacity cycle; infeasible remainder under capacity.")
                            break

                        trip_idx += 1
                        total_cost += pos_cost
                        # print(f"[Trip {trip_idx}] (position) start={current_start} -> end={pos_route[-1]} | "
                        #       f"cost={pos_cost:.2f} | covered=0 | remaining={len(req_set_remaining)} | req edges = {req_set_remaining}")
                        trips.append({
                            "start": current_start,
                            "end": pos_route[-1],
                            "route": pos_route,
                            "cost": pos_cost,
                            "covered_edges": set(),
                        })
                        # we don't mark the target depot banned yet; next iteration will test it
                        current_start = pos_route[-1]
                        continue

                    _seen_backtracks.add(state_key)

                    if trips:
                        last = trips.pop()
                        total_cost -= last['cost']
                        req_set_remaining |= last['covered_edges']
                        current_start = last['start']

                        router_bt = MagneticFieldRouter(
                            graph=G,
                            dist=dist,
                            start_depot=current_start,
                            depots=depots,
                            capacity=MAX_VEHICLE_CAPACITY,
                            alpha=1.0, gamma=1.0
                        )
                        route_bt, cost_bt, _ = router_bt.find_route_with_magnetic_scoring(list(req_set_remaining), verbose=False)
                        route_edges_bt = [tuple(sorted((route_bt[i], route_bt[i+1]))) for i in range(len(route_bt)-1)]
                        covered_bt = {e for e in route_edges_bt if e in req_set_remaining}

                        if not covered_bt:
                            print("[Backtrack] Could not salvage previous trip; attempting positioning trip fallback...")
                            pos_route, pos_cost = plan_positioning_trip_to_certified_depot(
                                G, dist, depots, MAX_VEHICLE_CAPACITY, current_start, req_set_remaining, banned
                            )
                            if pos_route is None:
                                print("[Stop] No valid positioning depot within one capacity cycle; infeasible remainder.")
                                break
                            trip_idx += 1
                            total_cost += pos_cost
                            # print(f"[Trip {trip_idx}] (position) start={current_start} -> end={pos_route[-1]} | "
                            #       f"cost={pos_cost:.2f} | covered=0 | remaining={len(req_set_remaining)} | req edges = {req_set_remaining}")
                            trips.append({
                                "start": current_start,
                                "end": pos_route[-1],
                                "route": pos_route,
                                "cost": pos_cost,
                                "covered_edges": set(),
                            })
                            current_start = pos_route[-1]
                            continue

                        trip_idx += 1
                        req_set_remaining -= covered_bt
                        total_cost += cost_bt
                        # print(f"[Trip {trip_idx}] start={current_start} -> end={route_bt[-1]} | "
                        #       f"cost={cost_bt:.2f} | covered={len(covered_bt)} | remaining={len(req_set_remaining)} | req edges = {req_set_remaining}")
                        trips.append({
                            "start": current_start,
                            "end": route_bt[-1],
                            "route": route_bt,
                            "cost": cost_bt,
                            "covered_edges": covered_bt,
                        })
                        current_start = route_bt[-1]
                        continue

                # retarget succeeded
                trip_idx += 1
                req_set_remaining -= ret_covered if ret_covered is not None else set()
                total_cost += ret_cost if ret_cost is not None else 0
                if ret_route is not None:
                    # print(f"[Trip {trip_idx}] start={current_start} -> end={ret_route[-1]} | "
                    #     f"cost={ret_cost:.2f} | covered={len(ret_covered)} | remaining={len(req_set_remaining)} | req edges = {req_set_remaining}")
                    trips.append({
                        "start": current_start,
                        "end": ret_route[-1],
                        "route": ret_route,
                        "cost": ret_cost,
                        "covered_edges": ret_covered,
                    })
                    current_start = ret_route[-1]
                    continue

            # Normal covering trip
            trip_idx += 1
            req_set_remaining -= covered_now
            total_cost += trip_cost
            print(f"[Trip {trip_idx}] start={current_start} -> end={route[-1]} | "
                  f"cost={trip_cost:.2f} | covered={len(covered_now)} | remaining={len(req_set_remaining)} | req edges = {req_set_remaining}")

            trips.append({
                "start": current_start,
                "end": route[-1],
                "route": route,
                "cost": trip_cost,
                "covered_edges": covered_now,
            })
            current_start = route[-1]

        # Summary
        print("\n=== Multi-Trip Summary ===")
        print(f"Trips committed: {len(trips)}")
        print(f"Total cost (no standalone relocations): {total_cost:.2f}")
        print(f"Required edges initially: {len(req_edges)}")
        print(f"Remaining uncovered edges: {len(req_set_remaining)}")
        
        for i, t in enumerate(trips, 1):
            tag = "" if t['covered_edges'] else " (position)"
            print(f"Trip {i}{tag}: start={t['start']} end={t['end']} "
                  f"cost={t['cost']:.2f} steps={len(t['route'])-1} covered={len(t['covered_edges'])} "
                  f"route={t['route']}")
        assert len(req_set_remaining) <= 2
if __name__ == "__main__":
    run_intelligent_tuning_demo()
