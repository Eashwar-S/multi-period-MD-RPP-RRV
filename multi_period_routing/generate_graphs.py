"""
Generate 15 connected random graphs with:
  - Nodes: 8–15
  - Edges: 20–40
  - Edge weights: random integer in [1, 25]
  - Edge probabilities: random float in (0, 1)
  - Depots placed as in routing.py (dispersed, feasibility-aware)

Saves each graph as graphs/<i>.pickle and metadata as graphs/graph_params.npy.

Metadata per graph (stored as a list of dicts in the .npy file):
  1. vehicle_capacity  = 2 * max_edge_weight
  2. recharge_time     = 1.5 * vehicle_capacity
  3. depot_nodes       = list of depot node IDs
  4. time_interval     = 2 * sum_of_all_edge_weights
"""

import os
import pickle
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ── reproducibility ──────────────────────────────────────────────────────────
MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "graphs_small_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_GRAPHS   = 15
NODE_MIN, NODE_MAX  = 8,  15
EDGE_MIN, EDGE_MAX  = 15, 30
WEIGHT_MIN, WEIGHT_MAX = 1, 25
NUM_PERIODS = 4


# ── depot placement (mirrors routing.py main()) ───────────────────────────────
def place_depots(G, B, path_len):
    """
    Greedily place depots so that:
      - Every graph edge is feasible from at least one depot
        (i.e. dist(depot, u) + w(u,v) + dist(depot, v) <= B)
      - Each new depot is within distance B of an existing depot
        (keeps the depot sub-graph connected)
      - Depots are as dispersed as possible
    """

    def edge_feasible(u, v, dep):
        dist_to = min(path_len[dep][u], path_len[dep][v])
        return dist_to + G[u][v]['weight'] + dist_to <= B

    def get_uncovered_edges(deps):
        return [(u, v) for u, v in G.edges()
                if not any(edge_feasible(u, v, d) for d in deps)]

    depots = [random.choice(list(G.nodes()))]

    while True:
        uncovered = get_uncovered_edges(depots)
        if not uncovered:
            break

        best_candidate = None
        best_score = -1

        for cand in G.nodes():
            if cand in depots:
                continue
            min_dist_to_depots = min(path_len[cand][d] for d in depots)
            if min_dist_to_depots > B:
                continue
            edges_covered = sum(1 for u, v in uncovered if edge_feasible(u, v, cand))
            if edges_covered == 0:
                continue
            score = edges_covered * 1000 + min_dist_to_depots
            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is not None:
            depots.append(best_candidate)
        else:
            # fallback: step towards the first uncovered edge while staying connected
            target_u, target_v = uncovered[0]
            best_fallback = None
            min_dist = float('inf')
            for cand in G.nodes():
                if cand in depots:
                    continue
                if min(path_len[cand][d] for d in depots) <= B:
                    dt = min(path_len[cand][target_u], path_len[cand][target_v])
                    if dt < min_dist:
                        min_dist = dt
                        best_fallback = cand
            if best_fallback is not None:
                depots.append(best_fallback)
            else:
                break  # connected graph should never reach here

    # ── enforce minimum 2 depots ──────────────────────────────────────────────
    # If the greedy algorithm was satisfied by a single depot (it already covered
    # every edge), force-add a second one: choose the non-depot node that is
    # farthest from the existing depot while still within distance B so the
    # depot sub-graph remains connected.
    if len(depots) < 2:
        candidates = [n for n in G.nodes() if n not in depots]
        # prefer nodes within B (connected depot graph)
        within_B = [n for n in candidates if path_len[depots[0]][n] <= B]
        pool = within_B if within_B else candidates   # fallback: any node
        second = max(pool, key=lambda n: path_len[depots[0]][n])
        depots.append(second)

    return depots


# ── visualization ────────────────────────────────────────────────────────────
def visualize_graph(G, depots, graph_id, B, TR, T_interval, save_dir):
    """Draw graph with spring layout, depot highlights and edge labels (w, p)."""
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))

    # nodes
    non_depot = [n for n in G.nodes() if n not in depots]
    nx.draw_networkx_nodes(G, pos, nodelist=non_depot,
                           node_color='lightgray', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=depots,
                           node_color='green', node_size=500, label='Depot')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)

    # edge labels:  w=<weight>  p=<prob>
    edge_labels = {
        (u, v): f"w={d['weight']}\np={d['prob']:.2f}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(
        f"Graph {graph_id}  |  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}\n"
        f"B={B:.0f}   TR={TR:.1f}   T_interval={T_interval:.0f}   depots={depots}",
        fontsize=10
    )
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{graph_id}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → saved visualisation: {out_path}")


# ── graph generation ──────────────────────────────────────────────────────────
metadata = []

for i in range(1, NUM_GRAPHS + 1):
    # draw a random (n, m) combination and ensure connectivity
    n = random.randint(NODE_MIN, NODE_MAX)
    m = random.randint(2*n, EDGE_MAX)

    seed = MASTER_SEED + i * 7   # deterministic but varied per graph
    attempts = 0
    while True:
        G = nx.gnm_random_graph(n, m, seed=seed)
        if nx.is_connected(G):
            break
        seed += 1
        attempts += 1
        if attempts > 500:
            # relax edge count slightly to guarantee connectivity
            m = min(m + 1, EDGE_MAX + 5)
            seed = MASTER_SEED + i * 7

    # assign edge attributes
    edge_probs_dict = {}
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(WEIGHT_MIN, WEIGHT_MAX)
        
        # generate NUM_PERIODS probabilities
        probs = [random.random() for _ in range(NUM_PERIODS)]
        G[u][v]['prob']   = probs[0]   # single edge probability for compat/viz
        
        # Canonical edge format
        e_canon = tuple(sorted((u, v)))
        edge_probs_dict[e_canon] = probs
        
    prob_path = os.path.join(OUTPUT_DIR, f"{i}_edge_probs.npy")
    np.save(prob_path, edge_probs_dict)

    # ── derived parameters ────────────────────────────────────────────────────
    max_w       = max(d['weight'] for _, _, d in G.edges(data=True))
    sum_w       = sum(d['weight'] for _, _, d in G.edges(data=True))

    B           = 2   * max_w          # vehicle capacity
    TR          = 1.5 * B              # recharge time
    T_interval  = 2   * sum_w          # time interval

    # ── depot placement ───────────────────────────────────────────────────────
    path_len = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    depots   = place_depots(G, B, path_len)

    # ── save graph ────────────────────────────────────────────────────────────
    pickle_path = os.path.join(OUTPUT_DIR, f"{i}.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)

    # ── record metadata ───────────────────────────────────────────────────────
    metadata.append({
        'graph_id':        i,
        'vehicle_capacity': B,
        'recharge_time':   TR,
        'depot_nodes':     depots,
        'time_interval':   T_interval,
    })

    print(f"Graph {i:2d}: nodes={G.number_of_nodes():2d}, edges={G.number_of_edges():2d}, "
          f"B={B:.0f}, TR={TR:.1f}, T={T_interval:.0f}, depots={depots}")
    visualize_graph(G, depots, i, B, TR, T_interval, OUTPUT_DIR)

# ── save metadata ─────────────────────────────────────────────────────────────
npy_path = os.path.join(OUTPUT_DIR, "graph_params.npy")
np.save(npy_path, metadata)
print(f"\nSaved {NUM_GRAPHS} graphs  →  {OUTPUT_DIR}/")
print(f"Saved metadata            →  {npy_path}")

# ── quick reload sanity check ─────────────────────────────────────────────────
loaded_meta = np.load(npy_path, allow_pickle=True)
assert len(loaded_meta) == NUM_GRAPHS, "Metadata count mismatch!"
print("Sanity check passed ✓")
