"""General Custom with functions and methods as parameters
Mod_type
Func 0 (Louvain): Default Directed Louvain
Func 2 (Cust 2): di*dj*Fr(j)/m
Func 6 (Cust 6): di*dj*log(1+Fr[j])
Func 11 (Cust 11): di/x^Fr[i] * dj*x^Fr[j]

FR_type
FR 0: FLOW
FR 1: FLOW_ng
FR 2: FLOW_ng_prop

FR_order = True: Order nodes by FR, False: Random order
FR_Recalc = True: Recalculate FR every iteration, False: Average FR every iteration
"""
import itertools
from collections import defaultdict, deque

import networkx as nx
import copy, random
from networkx.utils import py_random_state
import math
import numpy as np
import FlowRank_General as FR

from sklearn.metrics.cluster import normalized_mutual_info_score

DEBUG = False
#DEBUG = True

def log(s):
    if DEBUG:
        print(s)

def FlowRank_Func(edge_list,vlist,walk_len_c1,c_const=0,type=0):
    if type==0:
        return FR.FLOW(edge_list,vlist,walk_len_c1,c_const)
    elif type==1:
        return FR.FLOW_ng(edge_list,vlist,walk_len_c1,c_const)
    elif type==2:
        return FR.FLOW_ng_prop(edge_list,vlist,walk_len_c1,c_const)


def calc_FlowRank(graph, FR_type):
    node2FR = dict()
    if FR_type==3:
        pg_rank = nx.pagerank(graph,alpha=0.5)
        node2FR = {k: pg_rank[k]*graph.number_of_nodes() for k in pg_rank}
    else:
        for i in FlowRank_Func(graph.edges(),graph.nodes(),np.log2(graph.number_of_nodes()),0,FR_type):
            node_num = int(i[1])
            node2FR[node_num] = i[0]
    return node2FR

@py_random_state("seed")
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None, FR_order=False, FR_Recalc=False, FR_type=0, Mod_type=0, exp_base=6, init_part=None
):
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    #Calculate Flow Rank
    #node2FR = calc_FlowRank(graph, FR_type)
    node2FR = []

   

    #merge the induced subgraph nodes first
    if init_part is not None:
        #print('init_part:', init_part)
        graph = _gen_graph(graph, init_part)
        partition = init_part
    

    m = graph.size(weight="weight")

    print("# of Partition: ", len(partition))
    #inner part = partition of newly merged nodes
    #partition = partition of original graph nodes
    partition, inner_partition, improvement, total_improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, Mod_type, exp_base
    )
    print("# of Partition: ", len(partition))  
    # improvement = True
    total_improvement=threshold+1
    while total_improvement > threshold:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
          
        # #If we recalculate FR every iteration
        # if FR_Recalc:
        #     graph = _gen_graph(graph, inner_partition)
        #     #Calculate Flow Rank
        #     #node2FR = calc_FlowRank(graph, FR_type)
        # else: #If we just average FR every iteration
        #     graph, node2FR = _gen_graph_2(graph, inner_partition,node2FR)
        
        graph = _gen_graph(graph, inner_partition)
        
        partition, inner_partition, improvement, total_improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, Mod_type, exp_base
        )
        print("# of Partition: ", len(partition))

def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None, node2FR={}, FR_order=False, Mod_type=0,exp_base=6):
    node2com = {u: i for i, u in enumerate(G.nodes())}
    #print(node2com)
    inner_partition = [{u} for u in G.nodes()]

    #F_in and F_out are the general functions in the penalty term F_in(i)*F_out(j)/m in the modularity func
    #We can change F_in and F_out accordingly
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight")) #key = node, value = in_degree
        out_degrees = dict(G.out_degree(weight="weight")) #key = node, value = out_degree
        
        if Mod_type==0:
            F_in = {u: in_degrees[u] for u in G}
            F_out = {u: out_degrees[u] for u in G}
        elif Mod_type==2:
            F_in = {u: in_degrees[u] for u in G}
            F_out = {u: out_degrees[u]*node2FR[u] for u in G} #F_out(i) = FR(i)*out_degree(i)
        elif Mod_type==6:
            F_in = {u: in_degrees[u] for u in G}
            F_out = {u: out_degrees[u]*np.log2(1+node2FR[u]) for u in G}
        elif Mod_type==11:
            F_in = {u: in_degrees[u]/(exp_base**node2FR[u]) for u in G}
            F_out = {u: out_degrees[u]*(exp_base**node2FR[u]) for u in G}
        
        # print('F_in:',F_in)
        # print('F_out:',F_out)
        Stot_in = list(F_in.values()) #Each community's total incoming F(i)
        Stot_out = list(F_out.values()) #Each community's total outgoing F(i)
        # Calculate weights for both in and out neighbors without considering self-loops
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data="weight"):
                if u != n: 
                    nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt

    #Traversal Order
    if FR_order:
        node_list = sorted(G.nodes, key=lambda x: node2FR[x], reverse=True)
    else:
        node_list = list(G.nodes)
        seed.shuffle(node_list)
    
    nb_moves = 1
    improvement = False
    total_improvement=0
    while nb_moves > 0:
        nb_moves = 0
        for u in node_list:
            #best_mod = 0
            best_mod = 1e-7
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                Fin = F_in[u]
                Fout = F_out[u]
                Stot_in[best_com] -= Fin
                Stot_out[best_com] -= Fout
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (Fout * Stot_in[best_com] + Fin * Stot_out[best_com])
                    / m**2
                )
                
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            Fout * Stot_in[nbr_com]
                            + Fin * Stot_out[nbr_com]
                        )
                        / m**2
                    )
               
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += Fin
                Stot_out[best_com] += Fout
            
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
                total_improvement+=best_mod
                      
    partition = list(filter(len, partition)) # The partition of original graph nodes 
    inner_partition = list(filter(len, inner_partition)) # The partition of newly merged nodes in this round
    
    return partition, inner_partition, improvement, total_improvement

#Merge Nodes and create a new graph
def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H

#Merge nodes + Average FR to create a new graph
def _gen_graph_2(G, partition, node2FR):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    node2FR_new = {}

    for i, part in enumerate(partition):
        nodes = set()
        
        for node in part:
            #New node's FR is the average of all nodes in the community
            node2FR_new[i] = node2FR_new.get(i, 0) + node2FR[node]

            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
       
        #Average the FR of the community
        node2FR_new[i] /= len(part)
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H, node2FR_new

def _neighbor_weights(nbrs, node2com):
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbors as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights

def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H