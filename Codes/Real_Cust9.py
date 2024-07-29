"""Cust 9 (No closed form version)"""

import itertools
from collections import defaultdict, deque

import networkx as nx
import copy, random
from networkx.utils import py_random_state
import math
import numpy as np
import FlowRank_General as FR
DEBUG = False
#DEBUG = True

def log(s):
    if DEBUG:
        print(s)

def directed_modularity(G,partition,m):

    in_degrees = dict(G.in_degree(weight="weight"))
    out_degrees = dict(G.out_degree(weight="weight"))

    c_iden={}
    c=0
    Q=0
    for clusters in partition:

        for ell in clusters:
            c_iden[ell]=c

        c=c+1

    for (u,v) in G.edges():

        if(c_iden[u]==c_iden[v]):

            Q=Q+ (G[u][v]['weight']- out_degrees[u]*in_degrees[v]/(2*m))

    Q=(0.5/m)*Q

def cust9(G,node2com,m,u,c_num_new,inner_partition,node2FR,resolution=0.0):

    Q_c=0
    #Addition in new community
    for n in inner_partition[c_num_new]:
        if n==u:
            continue
        if G.has_edge(u,n):
            Q_c+=(G[u][n]['weight'])/m
        else:
            Q_c -= resolution*float(G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))*(node2FR[n])/(m*m)  
        if G.has_edge(n,u):
            Q_c+=(G[n][u]['weight'])/m
        else:
            Q_c -= resolution*float(G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))*(node2FR[u])/(m*m)
        # Q_c -= (G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))/(m*m)   
        # Q_c -= (G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))/(m*m)
       
    #Subtraction from old community
    for n in inner_partition[node2com[u]]:
        if n==u:
            continue
        if G.has_edge(u,n):
            Q_c-=(G[u][n]['weight'])/m
        else:
            Q_c += resolution*float(G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))*(node2FR[n])/(m*m)
        if G.has_edge(n,u):
            Q_c-=(G[n][u]['weight'])/m
        else:
            Q_c += resolution*float(G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))*(node2FR[u])/(m*m)
        # Q_c += (G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))/(m*m)
        # Q_c += (G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))/(m*m)

    return Q_c

def FlowRank_Func(edge_list,vlist,walk_len_c1,c_const=0,type=0):
    if type==0:
        return FR.FLOW(edge_list,vlist,walk_len_c1,c_const)
    elif type==1:
        return FR.FLOW_ng(edge_list,vlist,walk_len_c1,c_const)
    elif type==2:
        return FR.FLOW_ng_prop(edge_list,vlist,walk_len_c1,c_const)


@py_random_state("seed")
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None, FR_order=False, FR_Recalc=False, FR_type=0, Mod_type=0, exp_base=2
):
    
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    # mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    #Calculate Flow Rank
    node2FR = dict()
    for i in FlowRank_Func(graph.edges(),graph.nodes(),np.log2(graph.number_of_nodes()),0,FR_type):
        node_num = int(i[1])
        node2FR[node_num] = i[0]
    
    m = graph.size(weight="weight")

    if Mod_type==7:
        partition, inner_partition, improvement, total_improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, 2, exp_base
        )
    elif Mod_type==8:
        partition, inner_partition, improvement, total_improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, 6, exp_base
        )
    else:
        partition, inner_partition, improvement, total_improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, Mod_type, exp_base
        )
    # improvement = True
    total_improvement=threshold+1
    while total_improvement > threshold:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        # new_mod = modularity(
        #     graph, inner_partition, resolution=resolution, weight="weight"
        # )
        # if new_mod - mod <= threshold:
        #     return
        # mod = new_mod

        #If we recalculate FR every iteration
        if FR_Recalc:
            graph = _gen_graph(graph, inner_partition)
            #Calculate Flow Rank
            node2FR = dict()
            for i in FlowRank_Func(graph.edges(),graph.nodes(),np.log2(graph.number_of_nodes()),0,FR_type):
                node_num = int(i[1])
                node2FR[node_num] = i[0]
        else: #If we just average FR every iteration
            graph, node2FR = _gen_graph_2(graph, inner_partition,node2FR)
        
        if Mod_type==7:
            resolution = 0.25 #resolution for Louvain
        if Mod_type==8:
            resolution = 0.25
        partition, inner_partition, improvement, total_improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed, node2FR, FR_order, Mod_type, exp_base
        )

def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None, node2FR={}, FR_order=False, Mod_type=0,exp_base=2):
    print("one_level")
    """Calculate one level of the Louvain partitions tree

    Parameters
    ----------
    G : NetworkX Graph/DiGraph
        The graph from which to detect communities
    m : number
        The size of the graph `G`.
    partition : list of sets of nodes
        A valid partition of the graph `G`
    resolution : positive number
        The resolution parameter for computing the modularity of a partition
    is_directed : bool
        True if `G` is a directed graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    """
    #nx.draw(G, with_labels=True)
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        
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
        #log("nbrs: "+ str(nbrs))
    else:
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    

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
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            # log('weights2com: '+str(weights2com))
            # if is_directed:
            #     in_degree = in_degrees[u]
            #     out_degree = out_degrees[u]
            # else:
            #     degree = degrees[u]
            for nbr_com, wt in weights2com.items():
                

                if is_directed:
                    #takes O(n) time
                    # new_partition = copy.deepcopy(inner_partition)
                    # new_partition[node2com[u]].remove(u)
                    # new_partition[nbr_com].add(u)
                    #partition_temp = copy.deepcopy(inner_partition)
                    # gain=(
                    #     directed_modularity(G, new_partition, weight="weight", resolution=resolution)
                    #     - directed_modularity(G, partition_temp, weight="weight", resolution=resolution)    
                    # )

                    ##Here we use the particular function.
                    gain = cust9(G,node2com,m,u,nbr_com,inner_partition,node2FR,resolution=resolution)
                    
                    # log('u: '+str(u))
                    # log('nbr_com: '+str(nbr_com))
                    # log('inner_partition: '+str(inner_partition))
                    # # log('m: '+str(m))
                    # log('u:'+str(u)+' nbr_com: '+str(inner_partition[nbr_com])+ ' gain: '+str(gain))
                else:
                    new_partition = copy.deepcopy(inner_partition)
                    new_partition[node2com[u]].remove(u)
                    
                    #add the node to the new community
                    new_partition[nbr_com].add(u)
                    #remove any empty set
                    #new_partition = list(filter(len, new_partition))
                    partition_temp = copy.deepcopy(inner_partition)
                    #partition_temp = list(filter(len, partition_temp))
                    # gain = (
                    #     modularity(G, new_partition, weight="weight", resolution=resolution)
                    #     - modularity(G, partition_temp, weight="weight", resolution=resolution) 
                    # )
                    gain = 0
                    # print('gain: ',gain)
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
                    # log('custom gain: '+str(best_mod))
            # if is_directed:
            #     # Stot_in[best_com] += in_degree
            #     # Stot_out[best_com] += out_degree
            # else:
            #     Stot[best_com] += degree
            if best_com != node2com[u]:
                # print('best_com: ',best_com)
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
                total_improvement+=best_mod
                
                #print("gain:", gain, "best_mod:", best_mod, "total_improvement:", total_improvement)
                #gain=0
            #print("Check",gain,u,inner_partition[nbr_com])
            
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    # print('inner_partition: ',inner_partition)
   
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