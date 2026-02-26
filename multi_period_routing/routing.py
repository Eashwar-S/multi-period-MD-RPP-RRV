import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

def clean_trip(t):
    if not t: return t
    res = [t[0]]
    for x in t[1:]:
        if x != res[-1]:
            res.append(x)
    return res

def trip_length(G, trip):
    l = 0
    t = clean_trip(trip)
    if not t: return 0
    for i in range(len(t)-1):
        l += G[t[i]][t[i+1]]['weight']
    return l

def get_regions(G, num_regions=4, seed=42):
    np.random.seed(seed)
    pos = nx.spring_layout(G, seed=seed)
    pts = np.array([pos[n] for n in G.nodes()])
    
    idx = np.random.choice(len(pts), num_regions, replace=False)
    centers = pts[idx]
    
    for _ in range(20):
        dists = np.linalg.norm(pts[:, None] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        for i in range(num_regions):
            if np.sum(labels == i) > 0:
                centers[i] = np.mean(pts[labels == i], axis=0)
                
    node_region = {n: labels[i] for i, n in enumerate(G.nodes())}
    edge_region = {}
    for u, v in G.edges():
        e_can = tuple(sorted((u, v)))
        edge_region[e_can] = min(node_region[u], node_region[v])
    return node_region, edge_region, pos

def get_candidates(G, p_e, edge_region, K=10, max_per_region=3):
    edges = list(G.edges())
    edges.sort(key=lambda e: p_e[tuple(sorted(e))], reverse=True)
    
    C = []
    region_counts = {r: 0 for r in range(4)}
    
    for e in edges:
        e_can = tuple(sorted(e))
        r = edge_region[e_can]
        if region_counts[r] < max_per_region and e_can not in C:
            C.append(e_can)
            region_counts[r] += 1
        if len(C) == K:
            break
            
    if len(C) < K:
        for e in edges:
            e_can = tuple(sorted(e))
            if e_can not in C:
                C.append(e_can)
            if len(C) == K:
                break
    return C

def compute_signature(G, p_e, edge_region, num_regions=4, Q=20):
    m_b = np.zeros(num_regions)
    for e in G.edges():
        e_can = tuple(sorted(e))
        m_b[edge_region[e_can]] += p_e[e_can]
    s = np.sum(m_b)
    m_hat = m_b / s if s > 0 else m_b
    sig = tuple(int(np.round(m * Q)) for m in m_hat)
    return sig, m_hat

def get_top_covered(structure, p_e, k=5):
    covered = set()
    for v_id, trips in structure.items():
        for t in trips:
            t_clean = clean_trip(t)
            for i in range(len(t_clean)-1):
                e = tuple(sorted((t_clean[i], t_clean[i+1])))
                covered.add(e)
    cov_list = list(covered)
    cov_list.sort(key=lambda e: p_e.get(e, 0), reverse=True)
    return cov_list[:k]

class MemoryBank:
    def __init__(self, tau=0.85, max_entries=5):
        self.entries = []
        self.tau = tau
        self.max_entries = max_entries
        
    def retrieve(self, sig, m_hat):
        for e in self.entries:
            if e['signature'] == sig:
                return e
                
        best_sim = -1
        best_entry = None
        for e in self.entries:
            sim = 1 - 0.5 * np.sum(np.abs(m_hat - e['mass_vector']))
            if sim >= self.tau and sim > best_sim:
                best_sim = sim
                best_entry = e
        return best_entry

    def store_or_update(self, sig, m_hat, structure, stats, p_e):
        for e in self.entries:
            if e['signature'] == sig:
                e['route_structure'] = copy.deepcopy(structure)
                e['stats'] = stats
                e['reuse_count'] += 1
                e['last_used'] = stats['interval']
                e['mass_vector'] = m_hat
                e['skeleton_edges'] = get_top_covered(structure, p_e, 5)
                return
                
        entry = {
            'signature': sig,
            'mass_vector': m_hat,
            'route_structure': copy.deepcopy(structure),
            'stats': stats,
            'reuse_count': 0,
            'last_used': stats['interval'],
            'skeleton_edges': get_top_covered(structure, p_e, 5)
        }
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            for e in self.entries:
                e['evict_score'] = (e['reuse_count'] + 1) * e['stats']['reward']
            self.entries.sort(key=lambda x: x['evict_score'], reverse=True)
            self.entries.pop()

def compute_reward(G, R, p_e):
    covered = set()
    for trips in R.values():
        for t in trips:
            tc = clean_trip(t)
            for i in range(len(tc)-1):
                covered.add(tuple(sorted((tc[i], tc[i+1]))))
    return sum(p_e.get(e, 0) for e in covered), covered

def BuildFromScratch(G, depots, vehicles, C, p_e, B, TR, T_interval):
    path_len = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    
    R = {i: [] for i in vehicles}
    covered = set()
    
    v_states = {v: {'curr_time': 0, 'curr_depot': depots[v], 'active': True} for v in vehicles}
    
    while True:
        # Find active vehicle with minimum current time
        active_vs = [v for v in vehicles if v_states[v]['active']]
        if not active_vs:
            break
            
        v = min(active_vs, key=lambda x: v_states[x]['curr_time'])
        curr_time = v_states[v]['curr_time']
        curr_depot = v_states[v]['curr_depot']
        
        trip_nodes = [curr_depot]
        trip_len = 0
        added_any = False
        
        while True:
            best_edge = None
            best_score = -1
            best_info = None
            
            for e_can in C:
                if e_can in covered: continue
                u, v_node = e_can
                
                for st, en in [(u, v_node), (v_node, u)]:
                    dist_to = path_len[trip_nodes[-1]][st]
                    edge_w = G[st][en]['weight']
                    
                    best_depot_dist = min(path_len[en][d] for d in depots)
                    best_d = min(depots, key=lambda d: path_len[en][d])
                    
                    inc = dist_to + edge_w
                    if trip_len + inc + best_depot_dist <= B:
                        # Check if fits in T_interval with proper TR
                        time_added = trip_len + inc + best_depot_dist
                        if len(R[v]) > 0:
                            time_added += TR
                            
                        if curr_time + time_added <= T_interval:
                            score = p_e[e_can] / max(0.1, inc)
                            if score > best_score:
                                best_score = score
                                best_edge = e_can
                                best_info = (st, en, best_d, inc)
                            
            if best_edge is None:
                break
                
            st, en, best_d, inc = best_info
            part = paths[trip_nodes[-1]][st] + [en]
            trip_nodes = clean_trip(trip_nodes[:-1] + part)
            trip_len += inc
            covered.add(best_edge)
            added_any = True
            
        if not added_any:
            v_states[v]['active'] = False
            continue
            
        best_d = min(depots, key=lambda d: path_len[trip_nodes[-1]][d])
        path_home = paths[trip_nodes[-1]][best_d]
        trip_nodes = clean_trip(trip_nodes[:-1] + path_home)
        
        if len(trip_nodes) > 1:
            R[v].append(trip_nodes)
            exact_l = trip_length(G, trip_nodes)
            time_added = exact_l
            if len(R[v]) > 1:  # Since we already appended, > 1 means it's not the first trip
                time_added += TR
            v_states[v]['curr_time'] += time_added
            v_states[v]['curr_depot'] = best_d
        else:
            v_states[v]['active'] = False
            
    return R

def WarmStartFromMemory(Entry, G, depots, C, p_e, B, TR, T_interval, vehicles):
    path_len = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    C_set = set(C)
    
    R = {}
    covered = set()
    
    for v_id, trips in Entry['route_structure'].items():
        new_trips = []
        for t in trips:
            t = clean_trip(t)
            kept_edges = []
            for i in range(len(t)-1):
                e = tuple(sorted((t[i], t[i+1])))
                # Only keep edges that are in C_set AND have a high enough score relative to traversal cost
                if e in C_set and p_e.get(e, 0) > 0.1:
                    kept_edges.append((t[i], t[i+1], e))
                    
            if not kept_edges:
                continue
                
            new_t = [t[0]]
            for st, en, e in kept_edges:
                p = paths[new_t[-1]][st]
                cand_add = p + [en]
                cand_t = clean_trip(new_t[:-1] + cand_add)
                test_full = clean_trip(cand_t[:-1] + paths[en][t[-1]])
                if trip_length(G, test_full) <= B:
                    new_t = cand_t
                    covered.add(e)
            new_t = clean_trip(new_t[:-1] + paths[new_t[-1]][t[-1]])
            new_trips.append(new_t)
        R[v_id] = new_trips

    for v_id, trips in R.items():
        for i, t in enumerate(trips):
            while True:
                best_ins = None
                best_score = -1
                
                for e_can in C:
                    if e_can in covered: continue
                    u, v_node = e_can
                    
                    for j in range(len(t)-1):
                        for st, en in [(u, v_node), (v_node, u)]:
                            dist_to = path_len[t[j]][st]
                            edge_w = G[st][en]['weight']
                            dist_from = path_len[en][t[j+1]]
                            
                            old_edge_w = path_len[t[j]][t[j+1]]
                            inc = dist_to + edge_w + dist_from - old_edge_w
                            
                            # Check interval constraint too
                            rt = sum(trip_length(G, tmpt) for tmpt in trips)
                            if trips:
                                rt += max(0, len(trips)-1)*TR
                                
                            if trip_length(G, t) + inc <= B and rt + inc <= T_interval:
                                score = p_e[e_can] / max(0.1, inc)
                                if score > best_score:
                                    best_score = score
                                    best_ins = (j, st, en, e_can)
                                    
                if best_ins is None:
                    break
                j, st, en, e_can = best_ins
                part = paths[t[j]][st] + paths[en][t[j+1]]
                t = clean_trip(t[:j] + part + t[j+2:])
                covered.add(e_can)
            trips[i] = t
            
    final_R = {}
    for v_id, trips in R.items():
        rt = 0
        final_trips = []
        for t in trips:
            while trip_length(G, t) > B and len(t) > 2:
                t = clean_trip(paths[t[0]][t[-1]])
            l = trip_length(G, t)
            
            time_added = l
            if len(final_trips) > 0:
                time_added += TR
                
            if rt + time_added <= T_interval and len(t) > 1:
                final_trips.append(t)
                rt += time_added
                
        if final_trips:
            final_R[v_id] = final_trips
            
    # Also append built-from-scratch routes to unused vehicles
    used_vehicles = set(final_R.keys())
    unused_vehicles = [v for v in vehicles if v not in used_vehicles]
    
    if unused_vehicles:
        # Keep track of what we covered in warm start
        covered_scratch = set()
        for trips in final_R.values():
            for t in trips:
                tc = clean_trip(t)
                for i in range(len(tc)-1):
                    covered_scratch.add(tuple(sorted((tc[i], tc[i+1]))))
                    
        R_scratch = BuildFromScratch(G, depots, unused_vehicles, [c for c in C if c not in covered_scratch], p_e, B, TR, T_interval)
        for v in unused_vehicles:
            if R_scratch.get(v):
                final_R[v] = R_scratch[v]

    return final_R

def FastImprove(G, R, C, p_e, B, TR, T_interval, depots):
    path_len = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    C_set = set(C)
    
    for _ in range(50): # Increased iterations to help unstick routes
        covered_locs = []
        covered = set()
        for v_id, trips in R.items():
            for i, t in enumerate(trips):
                for j in range(len(t)-1):
                    e = tuple(sorted((t[j], t[j+1])))
                    covered.add(e)
                    if e in C_set:
                        covered_locs.append((e, v_id, i, j))
                        
        if not covered_locs:
            break
            
        covered_locs.sort(key=lambda x: p_e.get(x[0], 0))
        rem_cand = covered_locs[0]
        rem_e, v_id, trip_idx, rem_j = rem_cand
        
        t = R[v_id][trip_idx]
        kept = []
        for k in range(len(t)-1):
            e = tuple(sorted((t[k], t[k+1])))
            if e in C_set and e != rem_e:
                kept.append((t[k], t[k+1]))
                
        new_t = [t[0]]
        for st, en in kept:
            new_t = clean_trip(new_t[:-1] + paths[new_t[-1]][st] + [en])
        new_t = clean_trip(new_t[:-1] + paths[new_t[-1]][t[-1]])
        
        best_ins = None
        best_gain = -1
        
        for e_can in C:
            if e_can in covered or e_can == rem_e: continue
            u_cand, v_cand = e_can
            
            for k in range(len(new_t)-1):
                for st, en in [(u_cand, v_cand), (v_cand, u_cand)]:
                    dist_to = path_len[new_t[k]][st]
                    edge_w = G[st][en]['weight']
                    dist_from = path_len[en][new_t[k+1]]
                    
                    old_edge_w = path_len[new_t[k]][new_t[k+1]]
                    inc = dist_to + edge_w + dist_from - old_edge_w
                    
                    if trip_length(G, new_t) + inc <= B:
                        gain = p_e[e_can] - p_e.get(rem_e, 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_ins = (k, st, en, e_can, inc)
                            
        if best_gain > 0 and best_ins is not None:
            k, st, en, e_can, inc = best_ins
            old_len = trip_length(G, t)
            new_len = trip_length(G, new_t) + inc
            
            rt = sum(trip_length(G, tmpt) for tmpt in R[v_id])
            if R[v_id]:
                rt += max(0, len(R[v_id])-1)*TR
                
            rt_new = rt - old_len + new_len
            
            if rt_new <= T_interval:
                part = paths[new_t[k]][st] + paths[en][new_t[k+1]]
                final_t = clean_trip(new_t[:k] + part + new_t[k+2:])
                R[v_id][trip_idx] = final_t

def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    while True:
        G = nx.gnm_random_graph(15, 40, seed=seed)
        # G = nx.gnm_random_graph(100, 200, seed=seed)
        if nx.is_connected(G):
            break
        seed += 1
        
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 25)
        G[u][v]['probs'] = [random.random() for _ in range(3)]
        
    max_w = max(d['weight'] for u, v, d in G.edges(data=True))
    B = 2 * max_w
    TR = 1.5 * B
    P = 6*B
    T_interval = 3*P
    
    # Create depots such that they are dispersed, form a connected graph (edges <= B), 
    # and all graph edges are feasible to traverse from at least one depot.
    path_len = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    
    def edge_feasible(u, v, dep):
        dist_to = min(path_len[dep][u], path_len[dep][v])
        return dist_to + G[u][v]['weight'] + dist_to <= B
        
    def get_uncovered_edges(deps):
        uncovered = []
        for u, v in G.edges():
            covered = False
            for d in deps:
                if edge_feasible(u, v, d):
                    covered = True
                    break
            if not covered:
                uncovered.append((u, v))
        return uncovered

    # Start with a random depot
    depots = [random.choice(list(G.nodes()))]
    
    while True:
        uncovered = get_uncovered_edges(depots)
        if not uncovered:
            break
            
        # We need a new depot that:
        # 1. Is within distance B of at least one existing depot (to keep the depot graph connected)
        # 2. Covers at least one uncovered edge
        # 3. Is as far as possible from existing depots (to disperse)
        
        best_candidate = None
        best_score = -1
        
        for cand in G.nodes():
            if cand in depots:
                continue
                
            # Check connectivity to existing depots
            min_dist_to_depots = min(path_len[cand][d] for d in depots)
            if min_dist_to_depots > B:
                continue
                
            # Count how many uncovered edges it can cover
            edges_covered = sum(1 for u, v in uncovered if edge_feasible(u, v, cand))
            if edges_covered == 0:
                continue
                
            # Score: combination of edges covered and distance from existing depots (dispersion)
            score = edges_covered * 1000 + min_dist_to_depots
            if score > best_score:
                best_score = score
                best_candidate = cand
                
        if best_candidate is not None:
            depots.append(best_candidate)
        else:
            # Fallback: if we can't find a node that both connects to existing depots AND covers a new edge,
            # we just add a node that steps towards an uncovered edge while staying connected.
            # Pick the first uncovered edge
            target_u, target_v = uncovered[0]
            
            # Find a node 'cand' such that dist(cand, depot) <= B and dist(cand, target) is minimized
            best_fallback = None
            min_dist_to_target = float('inf')
            
            for cand in G.nodes():
                if cand in depots: continue
                if min(path_len[cand][d] for d in depots) <= B:
                    dist_to_target = min(path_len[cand][target_u], path_len[cand][target_v])
                    if dist_to_target < min_dist_to_target:
                        min_dist_to_target = dist_to_target
                        best_fallback = cand
                        
            if best_fallback is not None:
                depots.append(best_fallback)
            else:
                break # Should not happen if graph is connected
                
    vehicles = list(range(len(depots)))
    
    print(f"B: {B}, TR: {TR}, P: {P}, T_interval: {T_interval}")
    print(f"Depots generated: {depots} (Total: {len(depots)})")
    node_region, edge_region, pos = get_regions(G, num_regions=4, seed=42)
    memory_bank = MemoryBank()
    
    K_val = min(50, len(G.edges()))
    max_per = min(15, K_val)

    # Initialize probabilities
    p_e = {tuple(sorted((u, v))): random.random() for u, v in G.edges()}

    for n in range(5):  # Let's run for 5 intervals to see the effect
        # We need to compute C_list freshly based on updated p_e
        # Make candidates large enough so we actually see new edges
        C_list = get_candidates(G, p_e, edge_region, K=K_val, max_per_region=max_per)
        sig, m_hat = compute_signature(G, p_e, edge_region)
        
        entry = memory_bank.retrieve(sig, m_hat)
        cache_hit = False
        
        if entry is not None:
            cache_hit = True
            # Warm start from memory, but it might drop edges that decayed
            # Pass ALL vehicles so if a vehicle route gets empty, it can be re-planned
            R = WarmStartFromMemory(entry, G, depots, C_list, p_e, B, TR, P, vehicles)
        else:
            R = BuildFromScratch(G, depots, vehicles, C_list, p_e, B, TR, P)
            
        FastImprove(G, R, C_list, p_e, B, TR, P, depots)
        
        reward, unique_cov = compute_reward(G, R, p_e)
        stats = {'interval': n, 'reward': reward, 'runtime': 0.1}
        memory_bank.store_or_update(sig, m_hat, R, stats, p_e)
        
        print(f"--- Interval {n+1} ---")
        print(f"Signature Sigma(n): {sig}")
        print(f"Cache hit: {cache_hit}")
        print(f"Total unique reward sum: {reward:.4f}")
        print(f"Unique edges covered: {len(unique_cov)}")
        
        for v_id in vehicles:
            trips = R.get(v_id, [])
            print(f"  Vehicle {v_id}:")
            print(f"    Number of trips: {len(trips)}")
            total_time = 0
            if trips:
                for i, t in enumerate(trips):
                    l = trip_length(G, t)
                    total_time += l
                    print(f"    Trip {i+1}: seq={t}, travel_length={l}")
                total_time += (len(trips)-1) * TR 
            print(f"    Total route time: {total_time} (Budget: {P})")
        print()
        
        # Update probabilities for next interval based on coverage
        for e in G.edges():
            e_can = tuple(sorted(e))
            if e_can in unique_cov:
                # Decrease probability if covered
                p_e[e_can] *= 0.1
            else:
                # Increase probability if not covered recently
                p_e[e_can] = min(1.0, p_e[e_can] + 0.3)
                
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=depots, node_color='cyan', node_size=700, node_shape='s')
        nx.draw_networkx_labels(G, pos)
        
        covered_edges = list(unique_cov)
        uncovered_edges = [e for e in G.edges() if tuple(sorted(e)) not in unique_cov]
        
        nx.draw_networkx_edges(G, pos, edgelist=uncovered_edges, edge_color='gray', style='dashed')
        nx.draw_networkx_edges(G, pos, edgelist=covered_edges, edge_color='red', width=2)
        
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            e_can = tuple(sorted((u, v)))
            edge_labels[(u, v)] = f"w={d['weight']}\np={p_e[e_can]:.2f}"
            
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"Interval {n+1} - Signature: {sig}")
        plt.savefig(f"interval_{n+1}.png")
        plt.close()

if __name__ == "__main__":
    main()
