from sklearn.decomposition import PCA
import pynndescent
import numpy as np


def dir_KNN_graph(X,kchoice,dimension=0):


    if(dimension>0):
        transformer = PCA(n_components=dimension)
        X= transformer.fit_transform(X)


    n=X.shape[0]
    index = pynndescent.NNDescent(X)
    index.prepare()
    kchoice1=kchoice+1
    neighbors = index.query(X,k=kchoice1)
    indices = neighbors[0]
    knn_list=indices[:,1:]
    knn_list=np.array(knn_list)



    checked=np.zeros((n))

    edge_list=[]
    vlist=[]

    hashmap={}
    c=0
    i=0

    for i in range(n):
        for j in range(kchoice):

            u=i
            v=knn_list[i,j]

            if (u,v) in hashmap:
                c=c
            else:
                edge_list.append((u,v))        
                hashmap[(u,v)]=1
                c=c+1

            if(checked[u]==0):
                vlist.append(u)
            
            if(checked[v]==0):
                vlist.append(v)

            checked[u]+=1
            checked[v]+=1

        i=i+1

    return edge_list,vlist


def get_SNN_graph(edge_list,n,kchoice,threshold=0.11):

    knn_list=[[] for _ in range(n)]
    rev_knn_list=[[] for _ in range(n)]
    for (u,v) in edge_list:
        knn_list[u].append(v)
        rev_knn_list[v].append(u)

    
    SNN_list=[]
    for u in range(n):

        for x in knn_list[u]:
            for v in rev_knn_list[x]:
                c_n=len(set(knn_list[u]).intersection(knn_list[v]))/kchoice
                SNN_list.append((u,v,c_n))


    return SNN_list

