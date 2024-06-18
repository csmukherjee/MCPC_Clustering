"""Function for detecting communities based on Louvain Community Detection
Algorithm"""

import itertools
from collections import defaultdict, deque

import networkx as nx
import copy
from networkx.utils import py_random_state
from networkx.algorithms.community.louvain import _neighbor_weights

def modularity(G, communities, weight="weight", resolution=1):
    mod = 0
    #loop through every community
    for community in communities:
        com_sum=0
        #loop through every pair of nodes in the community
        if len(community) > 0:
            for u, v in itertools.combinations(community, 2):
                #add the weight of the edge between the two nodes
                if G.has_edge(u,v):
                    com_sum+= (
                        G[u][v]['weight']
                    )
            for node in community:
                #add the weight of the self-loop
                #if there is a self loop
                if G.has_edge(node,node):
                    com_sum+= (
                        G[node][node]['weight']
                    )
        if com_sum >=2:
            mod+=1
    return mod

def directed_modularity(G, communities, weight="weight", resolution=1):
    mod = 0
    #loop through every community
    for community in communities:
        com_sum=0
        #loop through every pair of nodes in the community
        if len(community) > 0:
            for u, v in itertools.combinations(community, 2):
                #add the weight of the edge between the two nodes
                if G.has_edge(u,v):
                    com_sum+= (
                        G[u][v]['weight']
                    )
                if G.has_edge(v,u):
                    com_sum+= (
                        G[v][u]['weight']
                    )
            for node in community:
                #add the weight of the self-loop
                #if there is a self loop
                if G.has_edge(node,node):
                    com_sum+= (
                        G[node][node]['weight']
                    )
        if com_sum >=2:
            mod+=1
        # print('mod: ',mod)
    return mod
  
def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None):
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
    nx.draw(G, with_labels=True)
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    # print('size of inner_partition: ',len(inner_partition))
    # print('inner_partition: ',inner_partition)
    # print('edge weight from 0 to 1: ',G[0][1]['weight'])
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight"))
        out_degrees = dict(G.out_degree(weight="weight"))
        
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
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                
            else:
                degree = degrees[u]
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    #print('neighbor_com: ',nbr_com)
                    new_partition = copy.deepcopy(inner_partition)
                    # print('u: ',u)
                    # print('partition: ',new_partition)
                    # print('node2com[u]: ',node2com[u])
                    new_partition[node2com[u]].remove(u)
                    # print('node2com[u]: ',node2com[u])
                    new_partition[nbr_com].add(u)
                    # print('new_partition: ',new_partition)
                    partition_temp = copy.deepcopy(inner_partition)
                    gain = (
                        directed_modularity(G, new_partition, weight="weight", resolution=resolution)
                        - directed_modularity(G, partition_temp, weight="weight", resolution=resolution)    
                    )
                    # print('gain: ',gain)
                else:
                    #print('neighbor_com: ',nbr_com)
                    new_partition = copy.deepcopy(inner_partition)
                    # print('u: ',u)
                    # print('partition: ',new_partition)
                    # print('node2com[u]: ',node2com[u])
                    new_partition[node2com[u]].remove(u)
                    # print('node2com[u]: ',node2com[u])
                    #add the node to the new community
                    new_partition[nbr_com].add(u)
                    # print('new_partition: ',new_partition)
                    #remove any empty set
                    #new_partition = list(filter(len, new_partition))
                    partition_temp = copy.deepcopy(inner_partition)
                    #partition_temp = list(filter(len, partition_temp))
                    #print('new_partition: ',new_partition)
                    gain = (
                        modularity(G, new_partition, weight="weight", resolution=resolution)
                        - modularity(G, partition_temp, weight="weight", resolution=resolution) 
                    )
                    # print('gain: ',gain)
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
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
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    # print('inner_partition: ',inner_partition)
    return partition, inner_partition, improvement