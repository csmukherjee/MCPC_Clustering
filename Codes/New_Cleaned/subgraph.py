import FlowRank_General as FR
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sknetwork.ranking import PageRank
from scipy import sparse
import networkx as nx
import numpy as np
import metric as met
from llist import dllist
import matplotlib.pyplot as plt
import igraph as ig
'''
FUnctions for Top k% induced subgraph

'''             
def getInducedSubgraph(G, top_nodes): #G = original graph, k = pick top k percent, node2FR = node to FR value
    #Remove nodes also remove adjacent edges
    H = G.copy()
    for u in G.nodes:
        if u not in top_nodes:
            H.remove_node(u)
    return H

def FlowRank_Func(edge_list,vlist,walk_len_c1,c_const=0,type=0):
    if type==0:
        return FR.FLOW(edge_list,vlist,walk_len_c1,c_const)
    elif type==1:
        return FR.FLOW_ng(edge_list,vlist,walk_len_c1,c_const)
    elif type==2:
        return FR.FLOW_ng_prop(edge_list,vlist,walk_len_c1,c_const)

def networkX_to_adjMatrix(G):
    # Get a list of nodes and create a mapping to indices
    nodes = list(G.nodes())
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    # Build the adjacency matrix
    row_indices = []
    col_indices = []
    data = []

    for source, target in G.edges():
        row_indices.append(node_indices[source])
        col_indices.append(node_indices[target])
        data.append(1)

    num_nodes = len(nodes)
    adjacency_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)), shape=(num_nodes, num_nodes)
    )

    return node_indices, adjacency_matrix

def calc_FlowRank(graph, FR_type, walk_len_c1):
    node2FR = dict()
    if FR_type==4:
        # pg_rank = nx.pagerank(graph,alpha=0.85) #alpha = 0.85 is the default
        # node2FR = {k: pg_rank[k]*graph.number_of_nodes() for k in pg_rank}

        pagerank = PageRank(damping_factor=0.85)
        node_indices, adj_matrix = networkX_to_adjMatrix(graph)
        pagerank.fit(adj_matrix)
        scores = pagerank.scores_
        node2FR = {node: scores[idx] for node, idx in node_indices.items()}
    elif FR_type==3:
        pg_rank = nx.pagerank(graph,alpha=0.85) #alpha = 0.85 is the default
        node2FR = {k: pg_rank[k] for k in pg_rank}
    else:
        for i in FlowRank_Func(graph.edges(),graph.nodes(),walk_len_c1,0,FR_type):
            node_num = int(i[1])
            node2FR[node_num] = i[0]
    
    return node2FR

def part_to_compressed_label(partition,H,original_n): #partition to labels (Returns compressed labels)
    #Mapping node numbers to index
        
    label_1=[-1]*(original_n)
    c=0
    for sets in partition:
        for ell in sets:
            label_1[ell]=c
        
        c=c+1
    
    label_compressed = []
    #for i in sorted(H.nodes()):
    for i in H.nodes():
        if label_1[i] == -1: 
            print('Error: Node not found in partition')
            return None
        label_compressed.append(label_1[i])
    #print(label_compressed)
    return label_compressed

def part_to_full_label(partition, original_n, mapping = None): #partition to labels (Returns full labels)
    label_1=[-1]*(original_n)
    '''
    iGraph has different node numbering than networkx (With mapping)
    '''
    if mapping is not None:
        for i, community in enumerate(partition):
            for node in community:
                label_1[mapping[node]] = i   
    else:    
        c=0
        for sets in partition:
            for ell in sets:
                label_1[ell]=c
            
            c=c+1
    #print(label_compressed)

    return label_1

def networkx_to_igraph(G):
    I = ig.Graph(directed=True)
    I.add_vertices(G.number_of_nodes())
    name_mapping = list(G.nodes)
    reverse_mapping = {name_mapping[i]: i for i in range(len(name_mapping))}
    I.vs['name'] = name_mapping
    edges = [(reverse_mapping[u], reverse_mapping[v]) for u, v in G.edges]
    I.add_edges(edges)
    return I, name_mapping

def get_NMI2(H_label, label):
    n = len(label)
    label_compressed = []
    for i in range(n):
        if H_label[i] != -1:
            label_compressed.append(label[i])
    
    H_label_compressed = []
    for i in range(n):
        if H_label[i] != -1:
            H_label_compressed.append(H_label[i])

    nmi_ = NMI(H_label_compressed, label_compressed)   
    #nmi_ = NMI(H_label_compressed, label_compressed)
    #print('nmi: ',nmi_, 'node #: ',len(H_label_compressed))
    return nmi_
        

def get_Purity2(H_label, label):
    n = len(label)
    label_compressed = []
    for i in range(n):
        if H_label[i] != -1:
            label_compressed.append(label[i])
    
    H_label_compressed = []
    for i in range(n):
        if H_label[i] != -1:
            H_label_compressed.append(H_label[i])

    purity_ = met.purity_score(H_label_compressed, label_compressed)   
    #purity_ = met.purity_score(H_label_compressed, label_compressed)
    #print('purity: ',purity_, 'node #: ',len(H_label_compressed))
    return purity_


'''
FUnctions for Datasets and Output

'''        
import data_utils_ch as data_util

def data_to_graph(name, survive=0):
    #scRNA datasets
    if name in ['Zhengmix8eq']:
        edge_list,vlist,n,label=data_util.local_SCRNA(name)
        #print("Dataset names is ",name," |V|, |E| #clusters= ",n,len(edge_list),len(set(label)))

    #These are for the bulk-RNA datasets.
    elif name in ['mRNA','miRNA']:
        #for survive in [0,1]:
        edge_list,vlist,label,n=data_util.local_bulkRNA(name,survive)
        label=data_util.set_labels(label)
        #print("Dataset names is ",name," |V|, |E| #clusters= ",n,len(edge_list),len(set(label)))

    #These contain image and document data.
    #datanames=['FashionMNIST','MNIST','seeds','breast-cancer','Omniglot','bbc_news','20NewsGroups','biorxiv','big_patent']
    elif name in ['FashionMNIST','MNIST','seeds','breast-cancer','Omniglot','bbc_news','20NewsGroups_tfdif','biorxiv','big_patent']:
        edge_list, label=data_util.load_data(name,kchoice=10)
        n=len(label)
        vlist=[i for i in range(n)]
        label=data_util.set_labels(label)
        #print("Dataset names is ",name," |V|, |E| #clusters= ",n,len(edge_list),len(set(label)))


    #These are data of 4 popular directed graphs. 
    elif name in ['Cora','Cora full','Citeseer','Eu core']:
        edge_list,vlist,label,n,good_v=data_util.graph_database(name)
        label=data_util.set_labels(label)
        #print("Dataset names is ",name," |V|, |E| #clusters= ",n,len(edge_list),len(set(label)))
    
    G = nx.DiGraph(edge_list)
    return G, label

def write_out(data, name, max_nmi, max_purity):
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write(name+'\n')
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('   Max NMI = [' + str(round(max_nmi[0],3)) + ',' + str(round(max_nmi[1],3)) + ']' + ' res:= ' + str(round(max_nmi[2],3)) + '\n')
    
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('   Partition sizes: [')
    for i in max_nmi[4]:
        #print(len(i),end=' ')
        with open('./Results/'+data+'.txt', 'a') as f:
            f.write(str(len(i))+', ')
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('] ' + '# of comm: ' + str(len(max_nmi[4]))+ '\n')
    
    
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('   Max Purity = [' + str(round(max_purity[0],3)) + ',' + str(round(max_purity[1],3)) + ']' + ' res:= ' + str(round(max_purity[2],3)) + '\n')
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('   Partition sizes: [')
    for i in max_purity[4]:
        with open('./Results/'+data+'.txt', 'a') as f:
            f.write(str(len(i))+', ')
    with open('./Results/'+data+'.txt', 'a') as f:
        f.write('] ' + '# of comm: ' + str(len(max_purity[4]))+ '\n')

def label_to_partition(H_label):
    new_partition = defaultdict(set)
    for i, node_label_ in enumerate(H_label):
        new_partition[node_label_].add(i)
    del new_partition[-1]
    for i, node_label_ in enumerate(H_label):
        if node_label_ == -1:
            new_partition[len(new_partition)+1].add(i)
    return list(new_partition.values())
'''
Functions for strong majority voting

'''
from collections import Counter, defaultdict
import random
def vote(G, H_label, node):
    #check every outgoing edge of node and counter the majority vote
    if len(G.out_edges(node)) == 0:
        return -1
    vt = defaultdict(int)
    for i in G.out_edges(node):
        #print('i:',i)
        vt[H_label[i[1]]] += 1
    #get the most common label
    most_common_label, most_common_count = Counter(vt).most_common(1)[0]
    
    #Strong Majority Vote
    if most_common_count <= len(G.out_edges(node))/2:
        return -1
    else:
        return most_common_label
    
def calc_balancedness(selected_labels_dict, cluster_sizes):
    #loop through the selected labels and calculate the balancedness
    min_cluster = 2
    max_cluster = -1
    for cluster, cluster_size in cluster_sizes.items(): #key = cluster #, value = count of nodes
        cnt = selected_labels_dict[cluster]
        ratio = cnt/cluster_size
        #print('cluster:',cluster, 'ratio:',round(ratio,3), end=' /')
        if ratio < min_cluster:
            min_cluster = ratio
        if ratio > max_cluster:
            max_cluster = ratio
    #print('min_cluster:',round(min_cluster,4), 'max_cluster:', round(max_cluster,4))
    return min_cluster/max_cluster

def calc_preservation(selected_labels_dict, cluster_sizes, num_total):
    num_selected = sum(selected_labels_dict.values())

    ratio = 0
    for cluster, cluster_size in cluster_sizes.items(): #key = cluster #, value = count of nodes
        cnt = selected_labels_dict[cluster]
        ratio += min(cnt/cluster_size, num_selected/num_total)
    ratio = (ratio/num_selected)*(num_total/len(cluster_sizes))

    return ratio

def merge_by_vote(top_nodes, nodes_rest, H_label, G, label, selected_labels_dict, cluster_sizes, k):
    #initial_num_of_nodes = len(top_nodes) - int(k*G.number_of_nodes())
    initial_num_of_nodes = 0
    #print('Initial number of nodes:', initial_num_of_nodes)
    #print('type of initial_num_of_nodes:', type(initial_num_of_nodes))
    flag = 1
    cnt = 0
    n = len(top_nodes)
    
    H_label_compressed = []
    True_label_compressed = []
    for i in top_nodes:
        if H_label[i] != -1:
            H_label_compressed.append(H_label[i])
            True_label_compressed.append(label[i])
    
    NMI_List = [NMI(H_label_compressed, True_label_compressed)]*(initial_num_of_nodes+1)
    Purity_List = [met.purity_score(True_label_compressed,H_label_compressed)]*(initial_num_of_nodes+1)
    Balance_List = [calc_balancedness(selected_labels_dict, cluster_sizes)]*(initial_num_of_nodes+1)
    Preserv_List = [calc_preservation(selected_labels_dict, cluster_sizes, G.number_of_nodes())]*(initial_num_of_nodes+1)

    total_inEdge = 0
    for node in top_nodes:
        if H_label[node] != -1:
            total_inEdge += len(G.in_edges(node))
    InEdge_List = [total_inEdge]
 
    node_linked_list = dllist(nodes_rest)

    
    while(flag):
        flag = 0
        #Reset Traversal (If found a node to merge, start the loop over)    
        nd = node_linked_list.first
        while nd:
            
            node = nd.value
            nd_next = nd.next
            #Already has a community assigned (Skip)
            if H_label[node] != -1:
                nd = nd_next
                continue
            
            new_comm = vote(G, H_label, node)
            if new_comm != -1:
                H_label[node] = new_comm
                node_linked_list.remove(nd)
                H_label_compressed.append(new_comm)
                True_label_compressed.append(label[node])
                InEdge_List.append(InEdge_List[-1] + len(G.in_edges(node)))
                selected_labels_dict[label[node]] += 1
                flag=1
                cnt +=1
                #Calculate new NMI every 5% of nodes
                if cnt > n/20:
                    new_nmi = NMI(H_label_compressed, True_label_compressed)
                    new_purity = met.purity_score(True_label_compressed,H_label_compressed) 
                    NMI_List.append(new_nmi)
                    Purity_List.append(new_purity) 
                    Balance_List.append(calc_balancedness(selected_labels_dict, cluster_sizes))
                    Preserv_List.append(calc_preservation(selected_labels_dict, cluster_sizes, G.number_of_nodes()))
                    #print each entry of selected_labels_dict and cluster_sizes & compare
                    
                    cnt = 0
                else:
                    NMI_List.append(NMI_List[-1])
                    Purity_List.append(Purity_List[-1])
                    Balance_List.append(Balance_List[-1])
                    Preserv_List.append(Preserv_List[-1])
            nd = nd_next
            
        
    NMI_List[-1] = NMI(H_label_compressed, True_label_compressed)
    Purity_List[-1] = met.purity_score(True_label_compressed,H_label_compressed)
    Balance_List[-1] = calc_balancedness(selected_labels_dict, cluster_sizes)
    Preserv_List[-1] = calc_preservation(selected_labels_dict, cluster_sizes, G.number_of_nodes())
    return NMI_List, Purity_List, Balance_List, Preserv_List, InEdge_List


def get_labels(partition,n_s):
    #final_partition_1 = deque(partition, maxlen=1).pop()
    #print(final_partition_1)


    label_1=np.zeros((n_s))
    c=0
    for sets in partition:
        for ell in sets:
            label_1[ell]=c
        
        c=c+1

    return label_1



# def relabel_graph(H): #compress the node numberings 
#     mapping = dict(zip(H.nodes(), range(H.number_of_nodes())))
#     H = nx.relabel_nodes(H, mapping)
#     return H

# def check_if_has_edge(H, partition_):
#     for i in H.nodes():
#         for j in H.nodes():
#             if i!=j and H.has_edge(i,j):
#                 #find i in partition_
#                 for k in range(len(partition_)):
#                     if i in partition_[k]:
#                         if j not in partition_[k]:
#                             print('Edge between nodes:',i,j) #Edge between two different communities
#     return

# def make_graph(edge_list):
#     G = nx.Graph(edge_list)
#     return G