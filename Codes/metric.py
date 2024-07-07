from re import template
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
import pynndescent
from sklearn.decomposition import PCA
import sklearn


import community as community_louvain
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


goodnames={}
goodnames['Katz_score']='Katz Centrality'
goodnames['pagerank_5']='PageRank (0.5)'
goodnames['pagerank_85']='PageRank (0.85)'
goodnames['pagerank_99']='PageRank (0.99)'
goodnames['cores']='Onion decomposition'
goodnames['deg_centrality']='Degree centrality'

goodnames['calc_LOF']='LOF'
goodnames['calc_hdbscan']='HDBSCAN'

goodnames['New method']='New method'

#log n steps
#1 step

goodnames['FLOW_ng']='N-Rank '
goodnames['FLOW_ng_prop']='RN-Rank '
goodnames['FLOW_ng2hopsimple']='N2-Rank '
goodnames['FLOW']='FlowRank '





# names=['N-Rank (1 step)','N-Rank (log n step)','RN-Rank (1 step)','N2-Rank (1 step)','In degree','PageRank (0.5)','PageRank (0.85)','PageRank (0.99)','K-core ranking','Betweenness','Closeness','Katz','Original accuracy']


# colormap={}
# colormap['RN-Rank (1 step)']='darkolivegreen'
# colormap['N2-Rank (1 step)']='darkgreen'
# colormap['N-Rank (1 step)']='limegreen' #'springgreen'
# colormap['N-Rank (log n step)']='limegreen'
# colormap['In degree']='orange'
# colormap['PageRank (0.5)']='salmon'
# colormap['PageRank (0.85)']='firebrick'
# colormap['PageRank (0.99)']='darkred'
# colormap['K-core ranking']='darkviolet'
# colormap['Betweenness']='magenta'
# colormap['Closeness']='deeppink'
# colormap['Katz']='orchid'
# colormap['Original accuracy']='black'



def result_plot(xaxis,Yaxes,methods,xtitle,ytitle,ftitle,colors):

    fig, ax = plt.subplots(figsize=(6,4), dpi=200)

    tt=0
    for method in methods:

        plt.plot(xaxis, Yaxes[:,tt],c=colors[tt],label=goodnames[method])
        #print(Yaxes[:,tt])
        tt=tt+1

    
    plt.rcParams.update({'font.size': 9})
    #ax.legend(loc='best',fontsize=11)
    #ax.legend(loc='upper right', borderaxespad=0., fontsize="14", bbox_to_anchor=(1.6, 1))
    plt.xlabel(xtitle,fontsize=13)
    plt.ylabel(ytitle,fontsize=13)
    plt.title(ftitle,fontsize=13)

    #CR=['1','1.3','1.6','2.2','2.9','3.9','5.46','8.2','13.9','32']
    #CR=[1.17,2.07,5.10,11.02,15.58,23.11,25.51]
    
    #CR=['0.00', '0.09', '0.20', '0.29', '0.36', '0.47', '0.56', '0.66', '0.77', '0.86', '0.95']
    
    CR=['0.00', '0.14', '0.26', '0.42', '0.54', '0.64', '0.71', '0.81', '0.89', '0.91']
    
    ax.set_xticks(xaxis)
    ax.set_xticklabels(CR)


    plt.show()




def preserve_ratio(label,new_label):

    rel=1
    n1=len(label)
    n2=len(new_label)

    

    s1=Counter(label)
    s2=Counter(new_label)


    ratio=0

    for i in s1:
        if(i in s2):
            ratio=ratio+min((s2[i]/s1[i]),n2/(rel*n1))  

    ratio=(ratio/n2)*(rel*n1)/len(set(label))

    return ratio


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def Knn_acc(knn_list,n,kchoice,iden):


    gc=0
    tc=0

    for i in range(n):
        for j in range(kchoice):
            if(iden[i]==iden[int(knn_list[i][j])]):
                gc=gc+1
            
            tc=tc+1

    return gc/tc

def KNN_graph_acc(X,kchoice,dim,label,returns=0):
    if(dim>0):
        transformer = PCA(n_components=dim)
        X= transformer.fit_transform(X)


    n=X.shape[0]
    index = pynndescent.NNDescent(X)
    index.prepare()
    kchoice1=kchoice+1
    neighbors = index.query(X,k=kchoice1)
    indices = neighbors[0]
    knn_list=indices[:,1:]
    knn_list=np.array(knn_list)

    acc=Knn_acc(knn_list,n,kchoice,label)
    if(returns==1):
        return acc
    else:
        print("Accuracy of ",kchoice,"-NN graph is",'%.3f'%acc)





#Breakdown.
gaps=20
c=1


def balancedness(v_cover_orders,methods,n,label,colors,core_names,name='plot'):

    
    print(n,len(label),set(label))
    print("balancedness AUC:")

    hmap={}
    tt=0
    t=0
    for ell in set(label):
        hmap[ell]=tt
        tt=tt+1



    subset=[[] for _ in range(len(set(label)))]


    for i in range(n):
        subset[hmap[label[i]]].append(i)


    tt=0
    plt.figure(figsize=(6,4), dpi=200)

    for method in methods:

        v_cover_order=v_cover_orders[tt]
        tt=tt+1

        xset=(1/gaps)*np.array([i for i in range(1,gaps//c+1)]).astype(float)

        UB=[]

        for n1 in [int(t/gaps*n) for t in range(1,gaps//c+1)]:
            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)

            values=[]

            clean=0


            for ell in core_names:

                values.append(len(set(accept).intersection(set(subset[ell])))/len(set(subset[ell])))


            UB.append(min(values)/max(max(values),0.001))



        plt.plot(xset, UB,c=colors[t],label=goodnames[method])
        t=t+1
        #print(names[method],"Balancedness AUC=",'%.3f'%sklearn.metrics.auc(xset, UB))
        print('%.2f'%(sklearn.metrics.auc(xset, UB)/((1/c-(1/gaps)))),' &',end=' ')



    plt.rcParams.update({'font.size': 8})
    #plt.legend(loc='best')
    plt.legend(loc='upper right', borderaxespad=0., fontsize="14", bbox_to_anchor=(1.6, 1))
    plt.xlabel("Fraction of points",fontsize=14)
    plt.ylabel("Balancedness",fontsize=14)
    plt.title(name,fontsize=14)
    #plt.savefig('concentric-GMM-balancedness',figsize=(10,10), dpi=200)
    plt.show()


alpha=3

    

def accuracy(v_cover_orders,edge_list,methods,colors,n,label,name='plot'):

    plt.figure(figsize=(6,4), dpi=200)
    
    tt=0
    t=0

    nf=[]

    for method in methods:

        v_cover_order=v_cover_orders[tt]
        tt=tt+1

    

        NNacc=[]


        xset=0.05*np.array([i for i in range(1,gaps//c+1)]).astype(float)

        for n1 in [int(t/gaps*n) for t in range(1,gaps//c+1)]:


            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)
            check=np.zeros(n)
            for ell in accept:
                check[ell]=1

            gc=0
            tc=0

            for (u,v) in edge_list: 
            
                if(check[u]==1 and check[v]==1):
                
                    if(label[u]==label[v]):
                        gc=gc+1
                
                    tc=tc+1

        
            NNacc.append(gc/tc)
            #print("accs:",n1,tc,gc)



        plt.plot(xset, NNacc,c=colors[t],label=goodnames[method])
        #print(names[method],"Accuracy AUC=",'%.3f'%sklearn.metrics.auc(xset, NNacc))
        print('%.3f'%(sklearn.metrics.auc(xset, NNacc)/((1/c-(1/gaps)))),' &',end=' ')
        t=t+1
        nf.append(NNacc[alpha])
    
    #plt.rcParams.update({'font.size': 12})
    #plt.legend(loc='best',fontsize='13')
    #plt.axvline(x=0.33, color='black', linestyle='--',label='Periphery points from here')
    #plt.legend(loc='upper right', borderaxespad=0., fontsize="24", bbox_to_anchor=(1.5, 1))
    #plt.legend(loc='best')
    #plt.legend(loc='upper right')
    
    plt.xlabel("Fraction of points",fontsize="14")
    plt.ylabel("Intra-community edge fraction",fontsize="14")
    plt.title(name,fontsize=14)
    #print("Accuracy AUC values:")

    #ax.set_xticks(x + width, xlabels, rotation=45, fontsize="12")
    ax = plt.gca()

    bottom, top = plt.ylim()
    top=min(top,1)

    yticks=[bottom+ (top-bottom)*i/5 for i in range(6)]
    ylabels=[str('%.2f'%yticks[i]) for i in range(6)]

    xticks=[0,0.2,0.4,0.6,0.8,1]
    xlabels=['0','0.2','0.4','0.6','0.8','1']
    ax.set_xticks(xticks,xlabels,fontsize="14")
    ax.set_yticks(yticks,ylabels,fontsize="14")


    #plt.savefig('conctric-GMM-KNN',figsize=(10,10), dpi=200)
    plt.show()

    return nf





#Preservation ratio
def preservstion(v_cover_orders,methods,colors,n,label,name='plot'):

    print("Preservation calculation")
    plt.figure(figsize=(6,4), dpi=200)

    tt=0
    t=0

    pf=[]

    for method in methods:

        v_cover_order=v_cover_orders[tt]
        tt=tt+1
        
        p_ratio=[]

        xset=0.05*np.array([i for i in range(1,gaps//c+1)]).astype(float)

        for n1 in [int(t/gaps*n) for t in range(1,gaps//c+1)]:


            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)

            order=list(map(int, accept))

            label_Y=label[order]
            
            p_ratio.append(preserve_ratio(label,label_Y))


            #if(n1==int((gaps//2)/gaps*n)):
            #    print(names[method],Counter(label_Y))



        plt.plot(xset, p_ratio,c=colors[t],label=goodnames[method])
        t=t+1
        #print(names[method],"Preservation AUC=",'%.3f'%sklearn.metrics.auc(xset, p_ratio))
        print('%.3f'%(sklearn.metrics.auc(xset, p_ratio)/((1/c-(1/gaps)))),' &',end=' ')     

        pf.append(p_ratio[alpha])

    plt.rcParams.update({'font.size': 9})
    plt.legend(loc='best')
    plt.xlabel("Fraction of points",fontsize=12)
    plt.ylabel("Preservation ratio",fontsize=12)
    plt.title(name,fontsize=12)
    #plt.savefig('usps-7-9-noise1-purity',figsize=(10,10), dpi=200)
    plt.show()



    return pf



#The outcomes themselves
def outcomes(v_cover_orders,methods,n,colors,label,name='plot'):


    plt.figure(figsize=(6,4), dpi=200)

    tt=0
    t=0
    for method in methods:

        v_cover_order=v_cover_orders[tt]
        tt=tt+1

        xaxis=[i for i in range(n)]


        plt.plot(xaxis,v_cover_order[:,0]/max(v_cover_order[:,0]),c=colors[t],label=goodnames[method])
        t=t+1
    

    plt.rcParams.update({'font.size': 9})
    plt.legend(loc='best')
    plt.xlabel("Points",fontsize=12)
    plt.ylabel("Algorithm value",fontsize=12)
    plt.title(name,fontsize=12)
    #plt.savefig('usps-7-9-noise1-purity',figsize=(10,10), dpi=200)
    plt.show()


##------ clustering accuracy----

def undir_KNN_graph(X,kchoice,dimension=0):
    

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
                edge_list.append((v,u))
                hashmap[(u,v)]=1
                hashmap[(v,u)]=1
                c=c+1

            if(checked[u]==0):
                vlist.append(u)
            
            if(checked[v]==0):
                vlist.append(v)

            checked[u]+=1
            checked[v]+=1

        i=i+1

    return edge_list,vlist



def LOUVAIN(edge_list,vlist,res=1):

    m=len(edge_list)

    G1= nx.Graph()
    G1.add_nodes_from(vlist)

    for i in range(m):
        u,v=edge_list[i]
        G1.add_edge(u,v)
                

    #print("Here now.")

    partition = community_louvain.best_partition(G1,resolution=res)
    #print(partition)
    res_iden=list(partition.values())

    return res_iden,partition


def Louvain_result(Y,label_Y,label):

    kchoice=20
    val=np.zeros(4)

    times=2
    for i in range(times):

        data1=Y
        labels=label_Y

        edge_list,vlist=undir_KNN_graph(data1,kchoice)
        label_1=[]
        for i in vlist:
            label_1.append(labels[i])


        new_label,partition=LOUVAIN(edge_list,vlist)



        nm1=normalized_mutual_info_score(label_1,new_label)
        ari1=adjusted_rand_score(label_1,new_label)
        ps1=purity_score(label_1,new_label)
        p_r1=preserve_ratio(label,label_1)

        val1=np.array([nm1,ari1,ps1,p_r1])

        val=val+val1


    return val/times


def louvain_plots(v_cover_orders,edge_list,PX,methods,n,label,name='plot'):

    print("Louvain clustering calculation")
    plt.figure(figsize=(6,4), dpi=200)

    gaps1=10
    tt=0
    final_val=[]
    for method in methods:

        print(goodnames[method])
        v_cover_order=v_cover_orders[tt]
        tt=tt+1

        res1=[]
        res2=[]
        
        p_ratio=[]

        

        #just set fixed percent
        for n1 in [n//5]:


            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)

            order=list(map(int, accept))

            label_Y=label[order]
            Y=PX[order,:]
            
            # val=Louvain_result(Y,label_Y,label)
            # print('.%2f'%(n1/n),"Subset", val,end=' ')

            # res1.append(val[0])
            
            check=np.zeros(n)
            for ell in accept:
                check[ell]=1

            gc=0
            tc=0
            edge_list1=[]
            vlist1=[]

            hmap={}

            for (u,v) in edge_list: 
            
                if(check[u]==1 and check[v]==1):

                    edge_list1.append((u,v))

                    if(u not in hmap):
                        vlist1.append(u)
                        hmap[u]=1

                    if(v not in hmap):
                        vlist1.append(v)
                        hmap[v]=1

            times=5
            val=np.zeros(4)
            label_1=label[vlist1]
            for ell in range(times):
                new_label,partition= LOUVAIN(edge_list1,vlist1)
                nm1=normalized_mutual_info_score(label_1,new_label)
                ari1=adjusted_rand_score(label_1,new_label)
                ps1=purity_score(label_1,new_label)
                p_r1=preserve_ratio(label,label_1)

                val1=np.array([nm1,ari1,ps1,p_r1])

                val=val+val1

            val=val/times

            res2.append(val[0])
            print("residual graph",val)

        final_val.append(val)

            


        xset=(1/gaps1)*np.array([i for i in range(4,gaps1//2+1)]).astype(float)
        #plt.plot(xset, res1,c=colormap[names[method]],label=names[method])
        #plt.plot(xset, res2,c=colormap[names[method]],label=names[method])
            


        
    val=Louvain_result(PX,label,label)
    print('Original',val)

    # plt.rcParams.update({'font.size': 9})
    # plt.legend(loc='best')
    # plt.xlabel("Points",fontsize=12)
    # plt.ylabel("NMI accuracy",fontsize=12)
    # plt.title(name,fontsize=12)
    #plt.savefig('usps-7-9-noise1-purity',figsize=(10,10), dpi=200)
    #plt.show()

    return final_val




def collective_auc(v_cover_orders,edge_list,methods,n,label,sub_label,core_names,sim=1,name='plot'):

    c=1
    plt.figure(figsize=(6,4))
    ttt=0
    gaps=20
    preserve_auc=[]
    NN_auc=[]
    Bval_auc=[]

    for method in methods:

        v_cover_order=v_cover_orders[ttt]
        ttt=ttt+1
        

        NN_acc=[]
        Bval=[]       

        xset=0.05*np.array([i for i in range(1,gaps+1)]).astype(float)

        #Balancedness-prep
        hmap={}
        temp=0
        for ell in set(sub_label):
            hmap[ell]=temp
            temp=temp+1
        subset=[[] for _ in range(len(set(sub_label)))]
        for i in range(n):
            subset[hmap[sub_label[i]]].append(i)
        #prep-end




        for n1 in [int(t/gaps*n) for t in range(1,gaps+1)]:


            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)
            order=list(map(int, accept))
            label_Y=label[order]
    
            
            #Balancedness
            values=[]
            for ell in core_names:

                values.append(len(set(accept).intersection(set(subset[ell])))/len(set(subset[ell])))


            Bval.append(min(values)/max(max(values),0.0001))


            #Accuracy
            check=np.zeros(n)
            for ell in accept:
                check[ell]=1

            gc=0
            tc=0
            ll=len(set(label))
            comp=np.zeros((ll,ll))
            
            for (u,v) in edge_list: 

                



            
                if(check[u]==1 and check[v]==1):

                    tu=label[u]
                    tv=label[v]
                    comp[tu,tv]=comp[tu,tv]+1
                
                    if(label[u]==label[v]):
                        gc=gc+1
                
                    tc=tc+1

            NN_acc.append(gc/tc)
            if(method==0 or method==2):

                for ll1 in range(ll):
                    comp[ll1,:]=comp[ll1,:]/sum(comp[ll1,:])
                #print(names[method],'\n',comp)
        
            
        
        nauc=sklearn.metrics.auc(xset[0:gaps//c], NN_acc[0:gaps//c])/((1/c-(1/gaps)))
        bauc=sklearn.metrics.auc(xset[0:gaps//c], Bval[0:gaps//c])/((1/c-(1/gaps)))
        print(goodnames[method],'%.3f'%nauc,'%.3f'%bauc)

        NN_auc.append(nauc)
        Bval_auc.append(bauc)

    plt.show()

    return NN_auc,Bval_auc




def collective_auc_real(v_cover_orders,edge_list,methods,n,label,name='plot'):

    

    ttt=0
    gaps=20
    NN_auc=[]
    Bval_auc=[]
    pauc_auc=[]
    c=1
    print("cvalue=",c)

    print("Accuracy   Preservation ratio Balancedness")
    for method in methods:

        v_cover_order=v_cover_orders[ttt]
        ttt=ttt+1
        

        NN_acc=[]
        Preserve=[]
        Bval=[]       

        xset=0.05*np.array([i for i in range(1,gaps+1)]).astype(float)

        #Balancedness-prep
        hmap={}
        temp=0
        for ell in set(label):
            hmap[ell]=temp
            temp=temp+1
        subset=[[] for _ in range(len(set(label)))]
        for i in range(n):
            subset[hmap[label[i]]].append(i)
        #prep-end




        for n1 in [int(t/gaps*n) for t in range(1,gaps+1)]:


            accept=np.array(v_cover_order[0:int(n1),1]).astype(int)
            order=list(map(int, accept))
            label_Y=label[order]


            #Preserve ratio.
            Preserve.append(preserve_ratio(label,label_Y))
    
            #Balancedness
            values=[]
            for ell in range(len(set(label))):

                values.append(len(set(accept).intersection(set(subset[ell])))/len(set(subset[ell])))


            Bval.append(min(values)/max(values))


            #Accuracy
            check=np.zeros(n)
            for ell in accept:
                check[ell]=1

            gc=0
            tc=0
            for (u,v) in edge_list: 
            
                if(check[u]==1 and check[v]==1):
                
                    if(label[u]==label[v]):
                        gc=gc+1
                
                    tc=tc+1

            NN_acc.append(gc/tc)

        
        tv=1/c
        nauc=sklearn.metrics.auc(xset[0:gaps//c], NN_acc[0:gaps//c])/((1/c-(1/gaps)))
        bauc=sklearn.metrics.auc(xset[0:gaps//c], Bval[0:gaps//c])/((1/c-(1/gaps)))
        pauc=sklearn.metrics.auc(xset[0:gaps//c], Preserve[0:gaps//c])/((1/c-(1/gaps)))
        print(goodnames[method],'%.3f'%nauc,'%.3f'%bauc,'%.3f'%pauc)

        NN_auc.append(nauc)
        Bval_auc.append(bauc)
        pauc_auc.append(pauc)

    
    

    return NN_auc,Bval_auc,pauc_auc





        


        

            








