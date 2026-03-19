
# Import Libraries 

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
import numpy as np
import os
import time 
import pandas as pd



def check_routes(m, x, numVehicle, numFlights, Tij, depotNodes):
    vals = m.getAttr('x', x)
    routes = {}
    edges_k = {}
    temp = True
    flag = False
    for k in range(0,numVehicle):
        for j in range(numFlights):
            routes[k,j] = []
            edges_k[k,j] = []


    for k, f, i, j in vals.keys():
        if vals[k, f, i, j] == 1.0:
            edges_k[k,f].append((i, j))


    for k in range(numVehicle):
        for f in range(numFlights):
            if edges_k[k,f]:
                l,edges = dfs(edges_k[k,f], depotNodes)
                if not l:
                    flag = True
                routes[k,f] = l
                c = 0
                for i in range(len(routes[k,f])-1):
                    c += Tij[routes[k,f][i],routes[k,f][i+1]]
                print(k,f,routes[k,f],c)
    if flag or edges:
        print('Routes are incorrect')
    else:
        print(edges)
        print('Routes are correct')
    return routes
   

class Node:
    def __init__(self, end_node, route, edge_list, parent):
        self.end_node = end_node
        self.route = route
        self.edge_list = edge_list
        self.parent = parent

def dfs(edges, depotNodes):
    node_list = []
    all_nodes = []
    for i,j in edges:
        if i in depotNodes:
            start_Node = Node(j,[i,j], edges.copy(), None)
            start_Node.edge_list.remove((i,j))
            edges.remove((i,j))
            node_list.append(start_Node)
            all_nodes.append(start_Node)
            break
        
    while node_list:
        current_node = node_list.pop(0)
        for i,j in current_node.edge_list:
            if current_node.end_node == i:
                new_Node = Node(j,current_node.route + [j], current_node.edge_list.copy(), current_node)
                new_Node.edge_list.remove((i,j))
                node_list.append(new_Node)
                all_nodes.append(node_list[-1])
                if (i,j) in edges:
                    edges.remove((i,j))
    
    all_nodes = all_nodes[::-1]
    
    route = []
    for node in all_nodes:
        if node.route[-1] in depotNodes and node.route[0] in depotNodes:
            if len(node.route) > len(route):
                route = node.route
    return route,edges


def find_subtours(edges, depot_nodes):
    """
    Given a list of directed edges for one (vehicle, flight), return a list of
    connected components (as sets of nodes) that do NOT contain any depot node.
    These are the illegal subtours that must be cut.
    """
    if not edges:
        return []

    # Build undirected adjacency for connectivity testing
    adj = {}
    nodes = set()
    for i, j in edges:
        nodes.add(i); nodes.add(j)
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    # BFS to find connected components
    visited = set()
    components = []
    for start in nodes:
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            queue.extend(adj.get(n, set()) - visited)
        components.append(comp)

    # Return only components that are entirely disconnected from depots
    return [c for c in components if not any(d in c for d in depot_nodes)]


def subtourelim_cb(model, where):
    """Gurobi lazy-constraint callback: cuts disconnected subtours per (k, f)."""
    if where != GRB.Callback.MIPSOL:
        return

    x_vars       = model._vars[0]
    numVehicle   = model._numVehicle
    numFlights   = model._numFlights
    depot_nodes  = model._depotNodes   # virtual depot node IDs

    vals = model.cbGetSolution(x_vars)

    for k in range(numVehicle):
        for f in range(numFlights):
            # Edges active in this (vehicle, flight)
            edges = [(i, j) for (kk, ff, i, j), v in vals.items()
                     if kk == k and ff == f and v > 0.5]

            subtours = find_subtours(edges, depot_nodes)

            for S in subtours:
                # Edges entirely within the subtour
                inner = [(i, j) for i, j in edges if i in S and j in S]
                if not inner:
                    continue
                # Classic subtour-elimination cut: sum(x within S) <= |S| - 1
                model.cbLazy(
                    gp.quicksum(x_vars[k, f, i, j] for i, j in inner)
                    <= len(S) - 1
                )

            


def edges_subset(requiredEdges, G, edgeWeight, depotNodes):
    dict_subset = {}
    for i,j in list(edgeWeight.keys()):
        # if i not in depotNodes and j not in depotNodes:
        edges = []
        neighbor = [p for p in G.neighbors(i) if p != j]
        for n in neighbor:
            edges.append((i,n))
            edges.append((n,i))
        neighbor = [p for p in G.neighbors(j) if p != i]
        for n in neighbor:
            edges.append((j,n))
            edges.append((n,j))
        dict_subset[i,j] = edges
        dict_subset[j,i] = dict_subset[i,j]
    return dict_subset

def subset_calculation(nodes):
    subset = []
    if len(nodes) <= 1:
        return subset
    for n in range(2, len(nodes)+1):
        l = combinations(nodes, n)
        for sub in l:
            subset.append(sub)
    return subset

def req_edges_in_subset(G, nodes, edgeWeight, requiredEdges):
    subset = subset_calculation(nodes)
    # print(subset)
    # print(len(subset))
    sub_edges = {}
    out_edges = {}
    in_edges = {}
    for sub in subset:
        sub_edges[sub] = []
        out_edges[sub] = []
        in_edges[sub] = []
        for i in range(len(sub)):
            for j in range(len(sub)):
                if [sub[i],sub[j]] in requiredEdges:
                    sub_edges[sub].append([sub[i],sub[j]])
                    sub_edges[sub].append([sub[j],sub[i]])
            
            neighbor = [p for p in G.neighbors(sub[i]) if p not in sub]
            for k in neighbor:
                out_edges[sub].append([sub[i], k])
                in_edges[sub].append([k, sub[i]])
        
    return sub_edges, out_edges, in_edges


def pre_processing(requiredEdges, depotNodes, edgeWeight, G):
    n = G.number_of_nodes() + 1
    for i, d_n in enumerate(depotNodes):
        G.add_edge(d_n, n, weight=0)
        edgeWeight[(d_n, n)] = 0
        G.add_edge(n, d_n, weight=0)
        edgeWeight[(n, d_n)] = 0
        depotNodes[i] = n
        n += 1    
    return edgeWeight, depotNodes, G


# def optimizationModel(G, numNodes, vehicles, flights, ij, numVehicle, numFlights, depotNodes,
#                       edgeWeight, B_k, Tij, vehicleCapacity, rechargeTime,
#                       edgeProbs, T_interval, I, J):
    
#     """
#     Maximize  prob - beta / T_interval   (probability coverage minus normalised makespan)

#     Where:
#       prob  = minimum per-vehicle probability score
#               = min_k ( sum_{e traversed by k} p_e )
#       beta  = makespan = max total mission time over all vehicles
#       T_interval = constant upper bound on beta

#     New decision variables:
#       c_e_k[(k, i, j)] in {0,1}  - 1 if vehicle k traverses undirected edge {i,j}
#       prob_k[k]  >=0              - probability score of vehicle k
#       prob       >=0              - lower bound (min) of prob_k over all k
#       beta       >=0              - makespan upper bound
#     """
#     m = gp.Model('MDRPPRRV_prob.lp')
#     Constant_M = 1e6

#     # ── canonical undirected edges ──────────────────────────────────────────
#     undirected_edges = list({(min(i,j), max(i,j)) for i,j in edgeWeight.keys()})

#     # ── decision variables ──────────────────────────────────────────────────
#     x      = m.addVars(vehicles, flights, ij, vtype=GRB.BINARY, name='x')
#     y_k    = m.addVars(numVehicle, numFlights, depotNodes, vtype=GRB.BINARY, name='y')
#     z      = m.addVars(numVehicle, numFlights, vtype=GRB.BINARY, name='z')

#     # Per-vehicle coverage indicator: c_e_k[k, i, j] = 1 if vehicle k covers edge {i,j}
#     c_e_k  = m.addVars(numVehicle, undirected_edges, vtype=GRB.BINARY, name='c_e_k')

#     # Per-vehicle probability score — sum of p_e for edges covered by vehicle k.
#     # Using SUM (not min) so covering more high-prob edges increases the score.
#     # The balance across vehicles is enforced at the vehicle level via min_k(prob_k).
#     prob_k = m.addVars(numVehicle, lb=0.0, ub=GRB.INFINITY,
#                        vtype=GRB.CONTINUOUS, name='prob_k')

#     # prob = min_k prob_k — ensures NO vehicle is used disproportionately
#     prob   = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='prob')

#     # beta = makespan (max total mission time across all vehicles)
#     beta   = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')

#     # ── ROUTING CONSTRAINTS (unchanged from original) ───────────────────────

#     # Depot outward flow: each vehicle must depart from its assigned depot on flight 0
#     m.addConstrs((z[k,0] == gp.quicksum(x[k, 0, i, j]
#                                         for i,j in list(edgeWeight.keys()) if i == B_k[k])
#                   for k in range(numVehicle)),
#                  name="Depot_Outward_Flow")

#     # Flight ordering: if vehicle k does not use flight f it cannot use flight f+1
#     m.addConstrs((z[k,f] - z[k,f+1] >= 0
#                   for k in range(numVehicle)
#                   for f in range(numFlights-1)),
#                  name="Flight_Order")

#     # Depot inward flow
#     m.addConstrs((gp.quicksum(x[k, f, i, j]
#                               for i,j in list(edgeWeight.keys()) if j == d) == y_k[k,f,d]
#                   for k in range(numVehicle)
#                   for f in range(numFlights)
#                   for d in depotNodes),
#                  name="Depot_Inward_Flow")

#     # Recharge / inter-flight: vehicle must land before next flight departs
#     m.addConstrs((y_k[k,f-1,d] >= gp.quicksum(x[k, f, i, j]
#                                                for i,j in list(edgeWeight.keys()) if i == d)
#                   for k in range(numVehicle)
#                   for f in range(1, numFlights)
#                   for d in depotNodes),
#                  name="Flight_Recharge")

#     # Each active flight lands at exactly one depot
#     m.addConstrs((z[k,f] - gp.quicksum(y_k[k,f,d] for d in depotNodes) == 0
#                   for k in range(numVehicle)
#                   for f in range(numFlights)),
#                  name="Flight_Land")

#     # Battery constraint: each single flight <= vehicleCapacity
#     m.addConstrs((gp.quicksum(x[k, f, i, j] * Tij[i, j]
#                               for i,j in list(edgeWeight.keys())) <= vehicleCapacity
#                   for k in range(numVehicle)
#                   for f in range(numFlights)),
#                  name="Battery")

#     # Makespan: total mission time (travel + recharge) <= beta for each vehicle
#     m.addConstrs(
#         (gp.quicksum(x[k, f, i, j] * Tij[i, j]
#                      for f in range(numFlights)
#                      for i,j in list(edgeWeight.keys()))
#          + (gp.quicksum(z[k,f] for f in range(numFlights)) - 1) * rechargeTime <= beta
#          for k in range(numVehicle)),
#         name="Makespan")

#     # Bound makespan to T_interval
#     m.addConstr(beta <= T_interval, name="Beta_TInterval")

#     # Depot flow balance
#     m.addConstrs(
#         (gp.quicksum(x[k, f, i, j] for i,j in list(edgeWeight.keys()) if i in depotNodes) ==
#          gp.quicksum(x[k, f, i, j] for i,j in list(edgeWeight.keys()) if j in depotNodes)
#          for k in range(numVehicle)
#          for f in range(numFlights)),
#         name="Depot_Flow_Balance")

#     # Flow conservation at non-depot nodes
#     # Use G.nodes() so 0-indexed nodes (including node 0) are included
#     all_nodes = list(G.nodes())
#     for k in range(numVehicle):
#         for f in range(numFlights):
#             for i in all_nodes:
#                 if i not in depotNodes:
#                     m.addConstr(
#                         gp.quicksum(x[k,f,i,j] for j in all_nodes if (k,f,i,j) in x) ==
#                         gp.quicksum(x[k,f,j,i] for j in all_nodes if (k,f,j,i) in x),
#                         name=f"Flow_Conservation_{k}_{f}_{i}")

#     # Flight starting: can only use edges on active flights
#     m.addConstrs(
#         (gp.quicksum(x[k,f,i,j] for i,j in list(edgeWeight.keys())) <= z[k,f] * Constant_M
#          for k in range(numVehicle)
#          for f in range(numFlights)),
#         name="Flight_Start")

#     # ── PROBABILITY COVERAGE CONSTRAINTS ────────────────────────────────────

#     # Linking: c_e_k[k, e] = 1 only if vehicle k traverses edge e (in either direction)
#     m.addConstrs(
#         (gp.quicksum(x[k, f, i, j] + x[k, f, j, i]
#                      for f in range(numFlights)) >= c_e_k[k, i, j]
#          for k in range(numVehicle)
#          for i,j in undirected_edges
#          if (i,j) in edgeWeight and (j,i) in edgeWeight),
#         name="Coverage_Link")

#     # Per-vehicle probability score = SUM of p_e for all edges covered by vehicle k.
#     # Covering more edges → higher prob_k → better objective.
#     # Taking min across vehicles (MinProb below) enforces equal workload sharing.
#     m.addConstrs(
#         (prob_k[k] == gp.quicksum(
#             edgeProbs.get((i, j), edgeProbs.get((j, i), 0.0)) * c_e_k[k, i, j]
#             for i, j in undirected_edges)
#          for k in range(numVehicle)),
#         name="ProbScore")


#     # prob <= prob_k[k] for all k  →  prob is lower bound (min) of per-vehicle scores
#     m.addConstrs(
#         (prob <= prob_k[k] for k in range(numVehicle)),
#         name="MinProb")

#     # ── OBJECTIVE ────────────────────────────────────────────────────────────
#     # Maximise minimum vehicle probability score minus normalised makespan
#     m.setObjective(prob - beta/T_interval, GRB.MAXIMIZE)
#     # m.setObjective(prob, GRB.MAXIMIZE)

#     m.reset()
#     m._vars       = [x, y_k, z, beta, prob, prob_k, c_e_k]
#     m._numVehicle = numVehicle
#     m._numFlights = numFlights
#     m._depotNodes = depotNodes   # virtual depot node IDs (used by callback)
#     m.setParam('PoolSolutions', 1)
#     m.Params.lazyConstraints = 1
#     m.optimize(subtourelim_cb)
#     return m, x


def optimizationModel(G, numNodes, vehicles, flights, ij, numVehicle, numFlights, depotNodes,
                      edgeWeight, B_k, Tij, vehicleCapacity, rechargeTime,
                      edgeProbs, T_interval, I, J,
                      lambda_unique=1.0,
                      lambda_overlap=0.25,
                      lambda_balance=0.5,
                      lambda_time=1.0):
    
    """
    Updated probabilistic MD-RPP-RRV MILP

    Objective:
        maximize
            lambda_unique  * UniqueExpectedCoverage
          - lambda_overlap * OverlapPenalty
          - lambda_balance * Imbalance
          - lambda_time    * (beta / T_interval)

    where
        UniqueExpectedCoverage = sum_e p_e * u_e
        OverlapPenalty         = sum_e p_e * overlap_e
        Imbalance              = P_max - P_min

    New / modified variables:
      c_e_k[k, e]   in {0,1} : 1 if vehicle k covers undirected edge e
      u_e[e]        in {0,1} : 1 if edge e is covered by at least one vehicle
      m_e[e]        >= 0     : number of vehicles covering edge e
      overlap_e[e]  >= 0     : duplicated coverage count = m_e - u_e
      prob_k[k]     >= 0     : expected hazard coverage assigned to vehicle k
      P_max, P_min  >= 0     : max/min vehicle probability scores
      beta          >= 0     : makespan upper bound

    Notes:
      - Unique coverage is rewarded through u_e
      - Duplicate coverage is penalized through overlap_e
      - Balance is enforced softly through (P_max - P_min)
      - Makespan is penalized through beta / T_interval
    """
    m = gp.Model('MDRPPRRV_prob_balanced_unique.lp')
    Constant_M = 1e6

    # ── canonical undirected edges ──────────────────────────────────────────
    undirected_edges = list({(min(i, j), max(i, j)) for i, j in edgeWeight.keys()})

    # probability lookup on canonical undirected edges
    p_e = {}
    for i, j in undirected_edges:
        p_e[(i, j)] = edgeProbs.get((i, j), edgeProbs.get((j, i), 0.0))

    # ── decision variables ──────────────────────────────────────────────────
    x   = m.addVars(vehicles, flights, ij, vtype=GRB.BINARY, name='x')
    y_k = m.addVars(numVehicle, numFlights, depotNodes, vtype=GRB.BINARY, name='y')
    z   = m.addVars(numVehicle, numFlights, vtype=GRB.BINARY, name='z')

    # Vehicle-edge coverage
    c_e_k = m.addVars(numVehicle, undirected_edges, vtype=GRB.BINARY, name='c_e_k')

    # Fleet-level unique coverage indicator
    u_e = m.addVars(undirected_edges, vtype=GRB.BINARY, name='u_e')

    # Number of vehicles covering each edge
    m_e = m.addVars(undirected_edges, lb=0.0, ub=numVehicle,
                    vtype=GRB.CONTINUOUS, name='m_e')

    # Duplicated/overlap coverage count: overlap_e = m_e - u_e
    overlap_e = m.addVars(undirected_edges, lb=0.0, ub=numVehicle,
                          vtype=GRB.CONTINUOUS, name='overlap_e')

    # Per-vehicle probability score
    prob_k = m.addVars(numVehicle, lb=0.0, ub=GRB.INFINITY,
                       vtype=GRB.CONTINUOUS, name='prob_k')

    # Soft fairness variables
    P_max = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_max')
    P_min = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_min')

    # Makespan
    beta = m.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')

    # ── ROUTING CONSTRAINTS (unchanged) ─────────────────────────────────────

    # Depot outward flow: each vehicle must depart from its assigned depot on flight 0
    m.addConstrs(
        (z[k, 0] == gp.quicksum(x[k, 0, i, j]
                                for i, j in list(edgeWeight.keys()) if i == B_k[k])
         for k in range(numVehicle)),
        name="Depot_Outward_Flow"
    )

    # Flight ordering
    m.addConstrs(
        (z[k, f] - z[k, f + 1] >= 0
         for k in range(numVehicle)
         for f in range(numFlights - 1)),
        name="Flight_Order"
    )

    # Depot inward flow
    m.addConstrs(
        (gp.quicksum(x[k, f, i, j]
                     for i, j in list(edgeWeight.keys()) if j == d) == y_k[k, f, d]
         for k in range(numVehicle)
         for f in range(numFlights)
         for d in depotNodes),
        name="Depot_Inward_Flow"
    )

    # Recharge / inter-flight continuity
    m.addConstrs(
        (y_k[k, f - 1, d] >= gp.quicksum(x[k, f, i, j]
                                         for i, j in list(edgeWeight.keys()) if i == d)
         for k in range(numVehicle)
         for f in range(1, numFlights)
         for d in depotNodes),
        name="Flight_Recharge"
    )

    # Each active flight lands at exactly one depot
    m.addConstrs(
        (z[k, f] - gp.quicksum(y_k[k, f, d] for d in depotNodes) == 0
         for k in range(numVehicle)
         for f in range(numFlights)),
        name="Flight_Land"
    )

    # Battery constraint
    m.addConstrs(
        (gp.quicksum(x[k, f, i, j] * Tij[i, j]
                     for i, j in list(edgeWeight.keys())) <= vehicleCapacity
         for k in range(numVehicle)
         for f in range(numFlights)),
        name="Battery"
    )

    # Makespan
    m.addConstrs(
        (gp.quicksum(x[k, f, i, j] * Tij[i, j]
                     for f in range(numFlights)
                     for i, j in list(edgeWeight.keys()))
         + (gp.quicksum(z[k, f] for f in range(numFlights)) - 1) * rechargeTime <= beta
         for k in range(numVehicle)),
        name="Makespan"
    )

    # Bound makespan
    m.addConstr(beta <= T_interval, name="Beta_TInterval")

    # Depot flow balance
    m.addConstrs(
        (gp.quicksum(x[k, f, i, j] for i, j in list(edgeWeight.keys()) if i in depotNodes) ==
         gp.quicksum(x[k, f, i, j] for i, j in list(edgeWeight.keys()) if j in depotNodes)
         for k in range(numVehicle)
         for f in range(numFlights)),
        name="Depot_Flow_Balance"
    )

    # Flow conservation at non-depot nodes
    all_nodes = list(G.nodes())
    for k in range(numVehicle):
        for f in range(numFlights):
            for i in all_nodes:
                if i not in depotNodes:
                    m.addConstr(
                        gp.quicksum(x[k, f, i, j] for j in all_nodes if (k, f, i, j) in x) ==
                        gp.quicksum(x[k, f, j, i] for j in all_nodes if (k, f, j, i) in x),
                        name=f"Flow_Conservation_{k}_{f}_{i}"
                    )

    # Can only use edges on active flights
    m.addConstrs(
        (gp.quicksum(x[k, f, i, j] for i, j in list(edgeWeight.keys())) <= z[k, f] * Constant_M
         for k in range(numVehicle)
         for f in range(numFlights)),
        name="Flight_Start"
    )

    # ── PROBABILITY / COVERAGE CONSTRAINTS (updated) ────────────────────────

    # c_e_k[k, e] = 1 only if vehicle k traverses undirected edge e in either direction
    for k in range(numVehicle):
        for i, j in undirected_edges:
            terms = []
            for f in range(numFlights):
                if (k, f, i, j) in x:
                    terms.append(x[k, f, i, j])
                if (k, f, j, i) in x:
                    terms.append(x[k, f, j, i])

            if terms:
                m.addConstr(gp.quicksum(terms) >= c_e_k[k, i, j],
                            name=f"Coverage_Link_{k}_{i}_{j}")
            else:
                m.addConstr(c_e_k[k, i, j] == 0,
                            name=f"Coverage_Link_Zero_{k}_{i}_{j}")

    # m_e[e] = number of vehicles covering edge e
    m.addConstrs(
        (m_e[i, j] == gp.quicksum(c_e_k[k, i, j] for k in range(numVehicle))
         for i, j in undirected_edges),
        name="Edge_Cover_Count"
    )

    # u_e[e] = 1 if at least one vehicle covers edge e
    m.addConstrs(
        (u_e[i, j] <= m_e[i, j]
         for i, j in undirected_edges),
        name="Unique_LB"
    )
    m.addConstrs(
        (m_e[i, j] <= numVehicle * u_e[i, j]
         for i, j in undirected_edges),
        name="Unique_UB"
    )

    # overlap_e[e] = m_e[e] - u_e[e]
    m.addConstrs(
        (overlap_e[i, j] == m_e[i, j] - u_e[i, j]
         for i, j in undirected_edges),
        name="Overlap_Def"
    )

    # Per-vehicle probability score
    m.addConstrs(
        (prob_k[k] == gp.quicksum(p_e[i, j] * c_e_k[k, i, j]
                                  for i, j in undirected_edges)
         for k in range(numVehicle)),
        name="ProbScore"
    )

    # Fairness spread variables
    m.addConstrs(
        (P_max >= prob_k[k] for k in range(numVehicle)),
        name="Pmax_Def"
    )
    m.addConstrs(
        (P_min <= prob_k[k] for k in range(numVehicle)),
        name="Pmin_Def"
    )

    # Optional: stop completely idle vehicle scores from making P_min artificially 0
    # if you want fairness only over used vehicles, uncomment and add an "active" linkage.
    # For now, this keeps the formulation simple and consistent with the current model.

    # ── OBJECTIVE ────────────────────────────────────────────────────────────
    unique_reward = gp.quicksum(p_e[i, j] * u_e[i, j] for i, j in undirected_edges)
    overlap_penalty = gp.quicksum(p_e[i, j] * overlap_e[i, j] for i, j in undirected_edges)
    imbalance_penalty = P_max - P_min
    time_penalty = beta / T_interval

    m.setObjective(
        lambda_unique * unique_reward,
        # - lambda_overlap * overlap_penalty
        # - lambda_balance * imbalance_penalty
        # - lambda_time * time_penalty,
        GRB.MAXIMIZE
    )

    # ── bookkeeping / solve ──────────────────────────────────────────────────
    m.reset()
    m._vars = [x, y_k, z, beta, prob_k, c_e_k, u_e, m_e, overlap_e, P_max, P_min]
    m._numVehicle = numVehicle
    m._numFlights = numFlights
    m._depotNodes = depotNodes
    m.setParam('PoolSolutions', 1)
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim_cb)

    return m, x

def print_and_visualize_solution(model, G_orig, depotNodes, edgeProbs, Tij_orig,
                                  vehicleCapacity, rechargeTime, T_interval,
                                  graph_id, numVehicle, numFlights, save_dir):
    """
    1. Print per-vehicle, per-flight route details to the terminal.
    2. Save one PNG per vehicle showing all its flights on the original graph.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    x_vars   = model._vars[0]   # x[k,f,i,j]
    prob_k_v = model._vars[5]   # prob_k[k]
    beta_val = model._vars[3].X
    prob_val = model._vars[4].X

    # spring layout on the original graph (without virtual depot nodes)
    pos = nx.spring_layout(G_orig, seed=42)

    # colour palette for flights
    flight_colours = list(mcolors.TABLEAU_COLORS.values())

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"SOLUTION DETAILS  |  graph {graph_id}")
    print(f"  Objective  = {model.ObjVal:.6f}")
    print(f"  prob (min) = {prob_val:.6f}")
    print(f"  beta       = {beta_val:.2f}  (T_interval = {T_interval:.0f})")
    print(f"{'─'*60}")

    for k in range(numVehicle):
        pk = prob_k_v[k].X
        print(f"\n  Vehicle {k}  |  prob_score = {pk:.4f}")

        # collect traversed edges per flight
        vehicle_edges = {}   # flight -> list of directed (i,j)
        total_travel  = 0.0
        for f in range(numFlights):
            flight_edges = [(i, j)
                            for (kk, ff, i, j), var in x_vars.items()
                            if kk == k and ff == f and var.X > 0.5]
            vehicle_edges[f] = flight_edges
            if flight_edges:
                t = sum(Tij_orig.get((i, j), 0) for i, j in flight_edges)
                total_travel += t
                # reconstruct route order via simple chain-following
                route = _chain_edges(flight_edges)
                route_str = ' → '.join(str(n) for n in route) if route else str(flight_edges)
                print(f"    Flight {f}: {route_str}")
                print(f"      edges    : {flight_edges}")
                print(f"      travel t : {t:.1f}  (capacity = {vehicleCapacity:.0f})")
                covered_probs = [edgeProbs.get((min(i,j), max(i,j)), 0.0)
                                 for i, j in flight_edges
                                 if (min(i,j), max(i,j)) in edgeProbs]
                print(f"      edge prob: {[round(p,3) for p in covered_probs]}")
            else:
                print(f"    Flight {f}: (idle)")

        num_active = sum(1 for f in range(numFlights) if vehicle_edges[f])
        mission_t  = total_travel + max(0, num_active - 1) * rechargeTime
        print(f"    Total mission time: {mission_t:.1f}")

        # ── visualise this vehicle ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))

        non_depot = [n for n in G_orig.nodes() if n not in depotNodes]
        nx.draw_networkx_nodes(G_orig, pos, nodelist=non_depot,
                               node_color='lightgray', node_size=500, ax=ax)
        nx.draw_networkx_nodes(G_orig, pos, nodelist=[d for d in depotNodes if d in G_orig.nodes()],
                               node_color='green', node_size=600, ax=ax, label='Depot')
        nx.draw_networkx_labels(G_orig, pos, ax=ax, font_size=9, font_weight='bold')

        # all edges in grey dashes (background)
        nx.draw_networkx_edges(G_orig, pos, edge_color='lightgray',
                               style='dashed', width=1.0, ax=ax)

        # edge labels (weight, prob)
        edge_labels = {(u, v): f"w={d['weight']}\np={d['prob']:.2f}"
                       for u, v, d in G_orig.edges(data=True)}
        nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=edge_labels,
                                     font_size=6, ax=ax)

        # traversed edges per flight with distinct colours
        for f in range(numFlights):
            fedges = vehicle_edges[f]
            if not fedges:
                continue
            # keep only edges that exist in the original graph (ignore virtual depot edges)
            orig_fedges = [(i, j) for i, j in fedges
                           if G_orig.has_edge(i, j) or G_orig.has_edge(j, i)]
            col = flight_colours[f % len(flight_colours)]
            nx.draw_networkx_edges(G_orig, pos, edgelist=orig_fedges,
                                   edge_color=col, width=3.0,
                                   arrows=True, arrowsize=20,
                                   connectionstyle='arc3,rad=0.1', ax=ax,
                                   label=f'Flight {f}')

        ax.set_title(
            f"Graph {graph_id} — Vehicle {k}\n"
            f"prob_score={pk:.4f}  mission_time={mission_t:.1f}  "
            f"(B={vehicleCapacity:.0f}, TR={rechargeTime:.0f}, T={T_interval:.0f})",
            fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        plt.tight_layout()

        png_path = os.path.join(save_dir, f"graph{graph_id}_vehicle{k}.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f"    → saved: {png_path}")

    print(f"{'─'*60}")


def _chain_edges(edges, start_node=None):
    """
    Walk through directed edges, consuming each exactly once (Hierholzer-style).
    Returns an ordered node sequence that uses every edge.
    Handles repeated node visits (non-simple paths / Eulerian circuits).
    """
    if not edges:
        return []

    # Build adjacency with edge counts
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)

    if start_node is None:
        in_nodes  = {j for _, j in edges}
        out_nodes = [i for i, _ in edges if i not in in_nodes]
        start_node = out_nodes[0] if out_nodes else edges[0][0]

    # Hierholzer algorithm (handles Eulerian paths through all edges)
    stack  = [start_node]
    route  = []
    while stack:
        v = stack[-1]
        if adj[v]:
            u = adj[v].pop(0)
            stack.append(u)
        else:
            route.append(stack.pop())
    route.reverse()
    return route


def print_and_visualize_solution(model, G_orig, depotNodes, edgeProbs, Tij_orig,
                                  vehicleCapacity, rechargeTime, T_interval,
                                  graph_id, numVehicle, numFlights, save_dir,
                                  virtual_to_real=None):
    """
    1. Print per-vehicle, per-flight route details to the terminal.
    2. Save one PNG per vehicle showing all its flights on the original graph.

    virtual_to_real: dict {virtual_node_id: real_depot_node_id}
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if virtual_to_real is None:
        virtual_to_real = {}
    real_to_label = {r: f'D{r}' for r in depotNodes}   # e.g. depot 6 → 'D6'
    virt_nodes    = set(virtual_to_real.keys())

    x_vars   = model._vars[0]   # x[k,f,i,j]
    prob_k_v = model._vars[4]   # prob_k[k]  (tupledict, index 4)
    beta_val = model._vars[3].X
    prob_val = min(prob_k_v[k].X for k in range(numVehicle))  # min across vehicles

    pos = nx.spring_layout(G_orig, seed=42)
    flight_colours = list(mcolors.TABLEAU_COLORS.values())
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"SOLUTION DETAILS  |  graph {graph_id}")
    print(f"  Objective  = {model.ObjVal:.6f}")
    print(f"  prob (min) = {prob_val:.6f}")
    print(f"  beta       = {beta_val:.2f}  (T_interval = {T_interval:.0f})")
    print(f"{'─'*60}")

    all_covered_edges_orig = set()

    for k in range(numVehicle):
        pk = prob_k_v[k].X
        # home depot of this vehicle
        home_virt  = model._vars[6]   # c_e_k — not needed here; B_k passed separately
        print(f"\n  Vehicle {k}  |  prob_score = {pk:.4f}")

        vehicle_orig_edges = {}   # flight → list of (i,j) edges in ORIGINAL graph only
        total_travel = 0.0

        for f in range(numFlights):
            # All edges the solver activated for (k, f)
            raw_edges = [(i, j)
                         for (kk, ff, i, j), var in x_vars.items()
                         if kk == k and ff == f and var.X > 0.5]

            if not raw_edges:
                vehicle_orig_edges[f] = []
                print(f"    Flight {f}: (idle)")
                continue

            # ── Identify departure depot ─────────────────────────────────────
            # The outgoing virtual-depot edge is (virt, real_depot)
            depart_virt = next(
                (i for i, j in raw_edges if i in virt_nodes), None)
            depart_real = virtual_to_real.get(depart_virt, depart_virt)

            # ── Identify arrival depot ───────────────────────────────────────
            arrive_virt = next(
                (j for i, j in raw_edges if j in virt_nodes), None)
            arrive_real = virtual_to_real.get(arrive_virt, arrive_virt)

            # ── Original-graph edges only (skip zero-weight virtual edges) ───
            orig_edges = [(i, j) for i, j in raw_edges
                          if i not in virt_nodes and j not in virt_nodes]
            vehicle_orig_edges[f] = orig_edges
            for u, v in orig_edges:
                all_covered_edges_orig.add((min(u, v), max(u, v)))

            # travel time uses only original edges
            t = sum(Tij_orig.get((i, j), Tij_orig.get((j, i), 0))
                    for i, j in orig_edges)
            total_travel += t

            # ── Reconstruct ordered route (real nodes + depot labels) ────────
            route_nodes = _chain_edges(orig_edges, start_node=depart_real)

            def _label(n):
                if n in virtual_to_real:
                    return f'D{virtual_to_real[n]}'
                return f'D{n}' if n in depotNodes else str(n)

            route_str = (f'D{depart_real} → '
                         + ' → '.join(_label(n) for n in route_nodes)
                         + f' → D{arrive_real}')

            # edge probs (original edges only, canonical)
            covered_probs = [edgeProbs.get((min(i,j), max(i,j)), 0.0)
                             for i, j in orig_edges
                             if (min(i,j), max(i,j)) in edgeProbs]

            print(f"    Flight {f}: {route_str}")
            print(f"      real edges   : {orig_edges}")
            print(f"      travel time  : {t:.1f}  (capacity = {vehicleCapacity:.0f})")
            print(f"      edge probs   : {[round(p,3) for p in covered_probs]}")

        num_active = sum(1 for f in range(numFlights) if vehicle_orig_edges.get(f))
        mission_t  = total_travel + max(0, num_active - 1) * rechargeTime
        print(f"    Total mission time: {mission_t:.1f}  "
              f"(beta = {beta_val:.1f}, T = {T_interval:.0f})")

        # ── Visualise ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))

        non_depot = [n for n in G_orig.nodes() if n not in depotNodes]
        nx.draw_networkx_nodes(G_orig, pos, nodelist=non_depot,
                               node_color='lightgray', node_size=500, ax=ax)
        nx.draw_networkx_nodes(G_orig, pos,
                               nodelist=[d for d in depotNodes if d in G_orig.nodes()],
                               node_color='green', node_size=650, ax=ax)
        # node labels: depots get 'D{n}', others get node id
        labels = {n: (f'D{n}' if n in depotNodes else str(n)) for n in G_orig.nodes()}
        nx.draw_networkx_labels(G_orig, pos, labels=labels, ax=ax,
                                font_size=9, font_weight='bold')

        # untraversed edges in light grey dashes
        traversed_set = {(min(i,j), max(i,j))
                         for f in range(numFlights)
                         for i, j in vehicle_orig_edges.get(f, [])}
        untraversed = [(u, v) for u, v in G_orig.edges()
                       if (min(u,v), max(u,v)) not in traversed_set]
        nx.draw_networkx_edges(G_orig, pos, edgelist=untraversed,
                               edge_color='lightgray', style='dashed', width=1.0, ax=ax)

        # edge labels (weight, prob)
        edge_labels = {(u, v): f"w={d['weight']}\np={d['prob']:.2f}"
                       for u, v, d in G_orig.edges(data=True)}
        nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=edge_labels,
                                     font_size=6, ax=ax)

        # traversed edges per flight — coloured by flight
        for f in range(numFlights):
            orig_fedges = vehicle_orig_edges.get(f, [])
            if not orig_fedges:
                continue
            col = flight_colours[f % len(flight_colours)]
            nx.draw_networkx_edges(G_orig, pos, edgelist=orig_fedges,
                                   edge_color=col, width=3.0,
                                   arrows=True, arrowsize=20,
                                   connectionstyle='arc3,rad=0.12', ax=ax,
                                   label=f'Flight {f}')

        ax.set_title(
            f"Graph {graph_id} — Vehicle {k}  (depot = D{virtual_to_real.get(None, '')})"
            f"\nprob_score={pk:.4f}  mission_time={mission_t:.1f}  "
            f"(B={vehicleCapacity:.0f}, TR={rechargeTime:.0f}, T={T_interval:.0f})",
            fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        plt.tight_layout()

        png_path = os.path.join(save_dir, f"graph{graph_id}_vehicle{k}.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f"    → saved: {png_path}")

    total_orig_edges = G_orig.number_of_edges()
    n_unique_covered = len(all_covered_edges_orig)
    pct_unique_covered = 100.0 * n_unique_covered / max(total_orig_edges, 1)

    print(f"  → Total Unique Edges Covered: {n_unique_covered} / {total_orig_edges} ({pct_unique_covered:.2f}%)")
    print(f"{'─'*60}")

    return pct_unique_covered


## GENERATED GRAPH INSTANCES ##
if __name__ == "__main__":

    import pickle

    graph_dir  = os.path.join(os.path.dirname(__file__), 'graphs')
    meta_all   = np.load(os.path.join(graph_dir, 'graph_params.npy'), allow_pickle=True)

    results = []

    # ── run against all 15 generated graphs ──────────────────────────────────
    for idx in range(len(meta_all)):
        if True:
            meta        = meta_all[idx]
            graph_id    = int(meta['graph_id'])
            if graph_id != 1:
                continue    
            vehicleCapacity = float(meta['vehicle_capacity'])
            rechargeTime    = float(meta['recharge_time'])
            T_interval      = float(meta['time_interval'])
            depotNodes      = list(meta['depot_nodes'])

            # load graph
            pkl_path = os.path.join(graph_dir, f'{graph_id}.pickle')
            with open(pkl_path, 'rb') as fh:
                G_orig = pickle.load(fh)

            print(f"\n{'='*60}")
            print(f"Graph {graph_id}: nodes={G_orig.number_of_nodes()}, "
                f"edges={G_orig.number_of_edges()}, "
                f"depots={depotNodes}")
            print(f"  B={vehicleCapacity}, TR={rechargeTime}, T={T_interval}")

            # build edgeWeight and edgeProbs (directed representation)
            edgeWeight = {}
            edgeProbs  = {}
            for u, v, d in G_orig.edges(data=True):
                edgeWeight[(u, v)] = d['weight']
                edgeWeight[(v, u)] = d['weight']
                canon = (min(u,v), max(u,v))
                edgeProbs[canon] = d['prob']

            # ── depot pre-processing: each depot gets a virtual node ──────────────
            G = G_orig.copy()
            new_depot_nodes = []
            n = G.number_of_nodes() + 1
            for d_n in depotNodes:
                G.add_edge(d_n, n, weight=0)
                edgeWeight[(d_n, n)] = 0
                G.add_edge(n, d_n, weight=0)
                edgeWeight[(n, d_n)] = 0
                new_depot_nodes.append(n)
                n += 1
            depotNodes_model = new_depot_nodes
            numNodes = G.number_of_nodes()

            # number of vehicles = number of depots (one per depot)
            numVehicle  = len(depotNodes_model)
            numFlights  = int(T_interval // (rechargeTime + vehicleCapacity))
            vehicles    = list(range(numVehicle))
            flights     = list(range(numFlights))

            # assign one vehicle per depot (round-robin if more vehicles than depots)
            B_k = [depotNodes_model[k % len(depotNodes_model)] for k in range(numVehicle)]

            ij  = {key: 0 for key in edgeWeight}
            Tij = dict(edgeWeight)
            I   = list({k[0] for k in edgeWeight})
            J   = list({k[1] for k in edgeWeight})

            sol_found = False
            # while numFlights <= 8:
            print(f'  Trying numFlights={numFlights} ...')
            start = time.time()
            model, x = optimizationModel(
                G, numNodes, vehicles, flights, ij, numVehicle, numFlights,
                depotNodes_model, edgeWeight, B_k, Tij, vehicleCapacity, rechargeTime,
                edgeProbs, T_interval, I, J)
            elapsed = time.time() - start

            if model.Status == GRB.OPTIMAL:# or model.Status == GRB.SUBOPTIMAL:
                obj       = model.ObjVal
                prob_k_v  = model._vars[4]   # tupledict: prob_k[k]
                beta_val  = model._vars[3].X
                prob_scores = [prob_k_v[k].X for k in range(numVehicle)]
                prob_val  = min(prob_scores)  # min across vehicles
                print(f'  SOLUTION FOUND in {elapsed:.2f}s')
                print(f'    Objective                 = {obj:.6f}')
                print(f'    prob_k (per-vehicle)      = {[round(p,6) for p in prob_scores]}')
                print(f'    prob (min vehicle prob)   = {prob_val:.6f}')
                print(f'    beta (makespan)           = {beta_val:.2f}')

                # ── Tij on original graph only (no virtual depot edges) ──
                Tij_orig = {(u, v): d['weight']
                            for u, v, d in G_orig.edges(data=True)}
                Tij_orig.update({(v, u): d['weight']
                                for u, v, d in G_orig.edges(data=True)})

                sol_dir = os.path.join(graph_dir, 'solutions')
                # build mapping: virtual depot node → real depot node
                virtual_to_real = {virt: real
                                for real, virt in zip(depotNodes, depotNodes_model)}
                pct_covered = print_and_visualize_solution(
                    model, G_orig, depotNodes, edgeProbs, Tij_orig,
                    vehicleCapacity, rechargeTime, T_interval,
                    graph_id, numVehicle, numFlights, sol_dir,
                    virtual_to_real=virtual_to_real)

                routes = check_routes(model, x, numVehicle, numFlights, Tij, depotNodes_model)
               
                np.save(os.path.join(graph_dir, f'routes_{graph_id}.npy'), routes)

                results.append({
                    'graph_id':   graph_id,
                    'num_nodes':  G_orig.number_of_nodes(),
                    'num_edges':  G_orig.number_of_edges(),
                    'num_depots': len(depotNodes),
                    'T_interval': T_interval,
                    'numVehicle': numVehicle,
                    'numFlights': numFlights,
                    'prob':       prob_val,
                    'beta':       beta_val,
                    'obj':        obj,
                    'time_s':     round(elapsed, 3),
                    '%_covered':  round(pct_covered, 2),
                })
                sol_found = True
                # break
            else:
                print(f'    No feasible solution — increasing flights')
                # numFlights += 1
                # flights = list(range(numFlights))

            if not sol_found:
                print(f'  No solution found within 8 flights.')
                results.append({
                    'graph_id':   graph_id,
                    'num_nodes':  G_orig.number_of_nodes(),
                    'num_edges':  G_orig.number_of_edges(),
                    'num_depots': len(depotNodes),
                    'T_interval': T_interval,
                    'numVehicle': numVehicle,
                    'numFlights': numFlights,
                    'prob':       None,
                    'beta':       None,
                    'obj':        None,
                    'time_s':     round(elapsed, 3),
                    '%_covered':  None,
                })

    # ── save summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv = os.path.join(graph_dir, 'optimisation_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
    print(df.to_string(index=False))
