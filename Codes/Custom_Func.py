"""Function for detecting communities based on Louvain Community Detection
Algorithm"""

import itertools
from collections import defaultdict, deque

import networkx as nx
import copy, random
from networkx.utils import py_random_state

#DEBUG = False
DEBUG = True

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

def update_directed_modularity(G,node2com,m,u,c_num_new,inner_partition):

    # out_neighbors=list(G.successors(u))
    # in_neighbors=list(G.predecessors(u))

    # log('u: '+str(u))
    # log('out_neighbors: '+str(out_neighbors))
    # log('in_neighbors: '+str(in_neighbors))
    Q_c=0
    #Addition in new community
    for n in inner_partition[c_num_new]:
        if n==u:
            continue
        if G.has_edge(u,n):
            Q_c+=(G[u][n]['weight'])/m
        if G.has_edge(n,u):
            Q_c+=(G[n][u]['weight'])/m
        Q_c -= (G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))/(m*m)
        Q_c -= (G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))/(m*m)
    #Subtraction from old community
    for n in inner_partition[node2com[u]]:
        if n==u:
            continue
        if G.has_edge(u,n):
            Q_c-=(G[u][n]['weight'])/m
        if G.has_edge(n,u):
            Q_c-=(G[n][u]['weight'])/m
        Q_c += (G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))/(m*m)
        Q_c += (G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))/(m*m)
    # for n in node2com[c_num_new]:
    #     if G.has_edge(u,n):
    #         Q_c+=(G[u][n]['weight'])/m
    #     if G.has_edge(n,u):
    #         Q_c+=(G[n][u]['weight'])/m
    #     Q_c -= (G.out_degree(u,weight='weight')*G.in_degree(n,weight='weight'))/(m*m)
    #     Q_c -= (G.out_degree(n,weight='weight')*G.in_degree(u,weight='weight'))/(m*m)
    # for v in out_neighbors:

    #     if(node2com[u]==node2com[v]):
    #         # Q_c=Q_c-( G[u][v]['weight']- G.out_degree(u,weight='weight')*G.in_degree(v,weight='weight')/(2*m) )
    #         Q_c=Q_c-( G[u][v]['weight']- G.out_degree(u,weight='weight')*G.in_degree(v,weight='weight')/(m) )/m
            
    #     if(node2com[v]==c_num_new):
    #         # Q_c=Q_c+( G[u][v]['weight']- G.out_degree(u,weight='weight')*G.in_degree(v,weight='weight')/(2*m) )
    #         Q_c=Q_c+( G[u][v]['weight']- G.out_degree(u,weight='weight')*G.in_degree(v,weight='weight')/(m) )/m

    # for v in in_neighbors:

    #     if(node2com[u]==node2com[v]):
    #         # Q_c=Q_c- ( G[v][u]['weight']- G.out_degree(v,weight='weight')*G.in_degree(u,weight='weight')/(2*m) )
    #         Q_c=Q_c- ( G[v][u]['weight']- G.out_degree(v,weight='weight')*G.in_degree(u,weight='weight')/(m) )/m

    #     if(node2com[v]==c_num_new):
    #         # Q_c=Q_c +( G[v][u]['weight']- G.out_degree(v,weight='weight')*G.in_degree(u,weight='weight')/(2*m) )
    #         Q_c=Q_c +( G[v][u]['weight']- G.out_degree(v,weight='weight')*G.in_degree(u,weight='weight')/(m) )/m

    return Q_c

def custom_directed_modularity(G,partition,m):
    


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

@py_random_state("seed")
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None
):
    """Yields partitions for each level of the Louvain Community Detection Algorithm

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The partitions at each level (step of the algorithm) form a dendrogram of communities.
    A dendrogram is a diagram representing a tree and each level represents
    a partition of the G graph. The top level contains the smallest communities
    and as you traverse to the bottom of the tree the communities get bigger
    and the overall modularity increases making the partition better.

    Each level is generated by executing the two phases of the Louvain Community
    Detection Algorithm.

    Be careful with self-loops in the input graph. These are treated as
    previously reduced communities -- as if the process had been started
    in the middle of the algorithm. Large self-loop edge weights thus
    represent strong communities and in practice may be hard to add
    other nodes to.  If your input graph edge weights for self-loops
    do not represent already reduced communities you may want to remove
    the self-loops before inputting that graph.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
     The name of an edge attribute that holds the numerical value
     used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities
    threshold : float, optional (default=0.0000001)
     Modularity gain threshold for each level. If the gain of modularity
     between 2 levels of the algorithm is less than the given threshold
     then the algorithm stops and returns the resulting communities.
    seed : integer, random_state, or None (default)
     Indicator of random number generation state.
     See :ref:`Randomness<randomness>`.

    Yields
    ------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.

    References
    ----------
    .. [1] Blondel, V.D. et al. Fast unfolding of communities in
       large networks. J. Stat. Mech 10008, 1-12(2008)

    See Also
    --------
    louvain_communities
    """

    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed
        )

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
    #nx.draw(G, with_labels=True)
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
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
        log("nbrs: "+ str(nbrs))
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    # random.seed(seed)
    # random.shuffle(rand_nodes)
    seed.shuffle(rand_nodes)
    log('rand_nodes: '+str(rand_nodes))
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            log('weights2com: '+str(weights2com))
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
                    gain = update_directed_modularity(G,node2com,m,u,nbr_com,inner_partition)
                    
                    # log('u: '+str(u))
                    # log('nbr_com: '+str(nbr_com))
                    # log('inner_partition: '+str(inner_partition))
                    # # log('m: '+str(m))
                    log('u:'+str(u)+' nbr_com: '+str(inner_partition[nbr_com])+ ' gain: '+str(gain))
                else:
                    new_partition = copy.deepcopy(inner_partition)
                    new_partition[node2com[u]].remove(u)
                    
                    #add the node to the new community
                    new_partition[nbr_com].add(u)
                    #remove any empty set
                    #new_partition = list(filter(len, new_partition))
                    partition_temp = copy.deepcopy(inner_partition)
                    #partition_temp = list(filter(len, partition_temp))
                    gain = (
                        modularity(G, new_partition, weight="weight", resolution=resolution)
                        - modularity(G, partition_temp, weight="weight", resolution=resolution) 
                    )
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

            #print("Check",gain,u,inner_partition[nbr_com])
            
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    # print('inner_partition: ',inner_partition)
   
    return partition, inner_partition, improvement

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