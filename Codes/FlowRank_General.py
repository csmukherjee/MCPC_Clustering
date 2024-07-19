


import random
from turtle import end_fill
from numba.typed import List
import warnings
from numba import njit
import pynndescent
import numpy as np
import operator
import matplotlib.pyplot as plt



#FLOWRank: random ascent.


def mat_flow_rank(adj_list,vlist,steps):

    n=len(vlist)
    k=len(adj_list[0])

    v_cover=np.ones((n))

    in_list=[[] for _ in range(n)]
    for i in range(n):
        for j in adj_list[i]:
            in_list[j].append(i)


    #Now change v_cover:
    for ell in range(steps):

        v_cover_n=np.zeros((n))
        for i in range(n):
            v_cover_n[i]=1/k*sum(v_cover[in_list[i]])

        v_cover=v_cover_n.copy()

    
    return v_cover






@njit
def T_PR(adj_list,walk_len_c1,c_const):

    n=len(adj_list)

    walk_len=walk_len_c1 #Changed here!


    v_cover=np.zeros((n))

    t=0
    for i in range(n):
        curr=i

        for j in range(c_const):
            k=len(adj_list[curr])
            pos=random.randint(0,k-1)
            t=t+1
            x=adj_list[curr][pos]
            curr=x



        for j in range(walk_len-c_const): 
            k=len(adj_list[curr])

            if(k<1):
                print("anarchy")
                break
        
            pos=random.randint(0,k-1)
            t=t+1
            x=adj_list[curr][pos]
            v_cover[x]=v_cover[x]+1
            curr=x

    return v_cover


#@njit
def mat_PR(mat_list,wt_list,walk_len_c1):

    n=len(mat_list)
    walk_len=int(np.log2(n)*walk_len_c1)

    vec=np.ones((n))


    for ell in range(walk_len):

        vec1=np.zeros(n)

        for i in range(n):
            t0=0
            for j in mat_list[i]:
                t0=t0+vec[j]/wt_list[j]
            
        vec=vec1


    return vec

@njit
def short_walk(adj_list,v_cover,v): 
#One round of random walk to higher FR neighbors, return the sequence of visited nodes & length
    vset=[v]
    times=1

    for times in range(times):
        curr=v
        stop=0
        t=0
        lent=0
        while(stop!=-1): 
            stepper=[]


            for ell in adj_list[curr]:

                if(v_cover[ell]>v_cover[curr]):
                     stepper.append(ell)

            k1=len(stepper)

            if(k1==0):
                stop=-1
            
            else:
                pos1=random.randint(0,k1-1)
                #print(v,lent,0,k1-1,pos1)
                curr=stepper[pos1]
                vset.append(curr)
                t=t+1
                

            lent+=1

        
    return vset,lent



def flow_calc(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)
    #print(n)


    v_cover=np.zeros(n)
    times1=200
    for ell in range(times1):
        v_cover1=T_PR(adj_list,walk_len_c1,c_const)
        v_cover=v_cover+v_cover1
  
    
    v_cover=v_cover/times1

    #v_cover=np.ones((n)) #test


    times=50

    rank=np.zeros((n))
    for v in vlist:

        sets=[v_cover[v]]
        for j in range(times):
            vset,lent=short_walk(adj_list,v_cover,v)
            #sets.append(max(vset))
            sets.append(v_cover[vset[-1]])
            # print('vset node:',vset[-1])
            # print('vset:',vset)
            # print('v_cover[vset]',v_cover[vset])
            # print('1:',rank[v]+ 1/((max(v_cover[vset]))/(v_cover[v])))
            #rank[v]=rank[v]+ 1/((max(v_cover[vset]))/(v_cover[v]))
        # print('v_cover[v]',v_cover[v])
        # print('sets:',sets)
        #print('2:',v_cover[v]/np.average(sets)) 
        if v_cover[v]==0:
            rank[v]=0
        else:  
            rank[v]=v_cover[v]/np.average(sets)
            
            

        
        #rank[v]=v_cover[v]/np.average(sets) #Edited this for checking.


    #reduce runtime later, just return rank

    xaxis=[i for i in range(n)]
    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis
    return v_cover_order

    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)

    # print(min(rank),max(rank))

    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order




def FLOW(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order


####--------------------------------


#Steepest ascent
#FLOW_MAX

@njit
def short_walk_max(adj_list,v_cover,v):

    vset=[]
    times=1

    for times in range(times):
        curr=v
        stop=0
        t=0
        lent=0

        while(stop!=-1):
            maxx=v_cover[curr]
            pos=-1

            for ell in adj_list[curr]:       
                if(v_cover[ell]>maxx):
                    maxx=v_cover[ell]
                    pos=ell


            if(pos==-1):
                stop=-1
            else:
                curr=pos
                vset.append(curr)

        
    return vset,lent



def flow_calc_max(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    mat=0
    #Using matrix multiplication.
    if(mat==1):
        mat_list1=[[] for i in range(n)]
        for i in range(len(adj_list)):
            for j in adj_list[i]:
                mat_list1[j].append(i)

        mat_list=mat_list1
#        mat_list=List(List(x) for x in mat_list1)
        wt_list=[len(adj_list[i]) for i in range(n)]
        wt_list=List(wt_list)
        v_cover=mat_PR(mat_list,wt_list,walk_len_c1)


    else:
        v_cover=T_PR(adj_list,walk_len_c1,c_const)

    
    v_cover=v_cover
    #print(v_cover.shape)
    xaxis=[i for i in range(n)]
    
    #Printing the first-rcound scores.
    #adj_len=[len(adj_list[i]) for i in range(n)]
    # plt.scatter(xaxis,v_cover,s=10)
    # plt.show()

    count=np.zeros((n))


    times=1

    rank=np.zeros((n))
    rank1=[]
    for v in vlist:
        for j in range(times):
            vset,lent=short_walk_max(adj_list,v_cover,v)
            
            if(len(vset)==0):
                rank1.append(1)
                rank[v]=1
            else:
                rank[v]=rank[v]+(v_cover[v]/max(v_cover[vset]))    
                rank1.append(v_cover[v]/max(v_cover[vset]))

            count[v]+=len(vset)

    count=count/times




    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis


    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)

    # print(min(rank),max(rank))
    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order


def FLOW_max(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_max(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order





###-------------------------------------------------------


# NeighborRank
def flow_calc_ng(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    v_cover=np.zeros(n)

    v_cover=mat_flow_rank(adj_list,vlist,int(walk_len_c1))


    # times1=200
    # for ell in range(times1):

    #     v_cover1=T_PR(adj_list,walk_len_c1,c_const)
    #     v_cover=v_cover+v_cover1



    xaxis=[i for i in range(n)]
    # plt.Figure()
    # plt.scatter(xaxis,v_cover,c='green',s=2)
    # plt.show()

    rank=np.zeros((n))

    for v in vlist:

        sc=0
        t=0
        sset=[]
        for ell in adj_list[v]:       
            if(v_cover[ell]>v_cover[v]):
            
              t=t+1
              sc=sc+v_cover[ell]
              #sc=sc+v_cover[v]/v_cover[ell]




        if(sc!=0 and t!=0):
            sc=sc/t
        else:
            sc=v_cover[v]

        # elif(sc!=0 and t==0):
        #     sc=1
        # else:        
        #      sc=v_cover[t]


        rank[v]=v_cover[v]/sc
        #print('1:',v_cover[v]/sc)
        #rank[v]=sc


    

    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis

    #print('1:',v_cover_order)
    return v_cover_order    
    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)
    #print('2:',v_cover_order)
    # print(min(rank),max(rank))

    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order
        
    


def FLOW_ng(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_ng(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order



##--------FLow_ng_naive


def flow_calc_ng_naive(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    v_cover=np.zeros(n)

    times1=1000
    for ell in range(times1):

        v_cover1=T_PR(adj_list,walk_len_c1,c_const)
        v_cover=v_cover+v_cover1



    xaxis=[i for i in range(n)]

    # plt.scatter(xaxis,v_cover,c='green',s=2)
    # plt.show()

    rank=np.zeros((n))

    for v in vlist:

        sc=0
        t=0
        sset=[]
        for ell in adj_list[v]:       
            t=t+1
            sc=sc+v_cover[ell]


        if(sc!=0 and t!=0):
            sc=sc/t
        else:
            sc=v_cover[v]


        rank[v]=v_cover[v]/sc


    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis


    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)


    return v_cover_order


def FLOW_ng_naive(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_ng_naive(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order



##-----------------------------------------------
## FLOW 2-hop-neighbor?

def flow_calc_ng2(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    v_cover=np.zeros(n)

    v_cover=mat_flow_rank(adj_list,vlist,int(walk_len_c1))

    # times1=200
    # for ell in range(times1):

    #     v_cover1=T_PR(adj_list,walk_len_c1,c_const)
    #     v_cover=v_cover+v_cover1



    xaxis=[i for i in range(n)]

    # plt.scatter(xaxis,v_cover,c='green',s=2)
    # plt.show()

    rank=np.zeros((n))

    for v in vlist:

        sc=0
        t=0
        sset=[]
        hmap={}
        for ell in adj_list[v]:

            if(v_cover[ell]>v_cover[v]):

                for ell1 in adj_list[ell]:

                    if(v_cover[ell1]>v_cover[ell]):

                        for ell2 in adj_list[ell1]:
                            
                            if(v_cover[ell2]>v_cover[ell1]):

                                if(ell1 in hmap):
                                    hmap[ell1]=hmap[ell1]+1
                                else:
                                    hmap[ell1]=1


        tmin=0
        for ell in hmap:
            if(hmap[ell]>tmin):
                tmin=hmap[ell]
            

        if(len(hmap)>0):
            for ell in hmap:
                if(hmap[ell]>=tmin):
                    sc=sc+(hmap[ell]*v_cover[ell])
                    t=t+hmap[ell]

        else:

            for ell in adj_list[v]:
                if(v_cover[ell]>v_cover[v]): 

                    sc=sc+v_cover[ell]
                    t=t+1





            # if(hmap[ell]>=min(tmin//2,tmin-1) and v_cover[ell]>v_cover[v]):
            #     t=t+1
                
            #     sc=sc+v_cover[ell]
                
            #     #sset.append(v_cover[ell])
            #     #sc=sc+v_cover[v]/v_cover[ell]




        if(sc!=0 and t!=0):
            sc=sc/t
        else:
            sc=v_cover[v]


        rank[v]=v_cover[v]/sc

        #rank[v]=sc


    

    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis


    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)

    # print(min(rank),max(rank))

    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order


def FLOW_ng2(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_ng2(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order


###----- Flow_ng_propagate


def flow_calc_ng_prop(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    v_cover_order1=flow_calc_ng(adj_list,vlist,walk_len_c1,c_const)

    v_cover_order1=np.array(v_cover_order1)

    v_cover_order2=sorted(v_cover_order1,key=operator.itemgetter(1),reverse=False)

    v_cover=np.array(v_cover_order2)[:,0]

    #print(v_cover.shape)



    xaxis=[i for i in range(n)]

    # plt.scatter(xaxis,v_cover,c='green',s=2)
    # plt.show()

    rank=np.zeros((n))

    for v in vlist:

        sc=0
        t=0
        sset=[]
        for ell in adj_list[v]:       
            if(v_cover[ell]>v_cover[v]):
            
              t=t+1
              sc=sc+v_cover[ell]
              #sc=sc+v_cover[v]/v_cover[ell]




        if(sc!=0 and t!=0):
            sc=sc/t
        else:
            sc=v_cover[v]

        # elif(sc!=0 and t==0):
        #     sc=1
        # else:        
        #      sc=v_cover[t]


        rank[v]=v_cover[v]/sc

        #rank[v]=sc


    

    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis
    return v_cover_order

    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)

    # print(min(rank),max(rank))

    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order


def FLOW_ng_prop(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_ng_prop(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order




##--------- 2 hop simple.
def flow_calc_ng2hopsimple(adj_list,vlist,walk_len_c1,c_const):
    
    n=len(adj_list)

    v_cover=np.zeros(n)

    times1=200
    for ell in range(times1):

        v_cover1=T_PR(adj_list,walk_len_c1,c_const)
        v_cover=v_cover+v_cover1



    xaxis=[i for i in range(n)]

    # plt.scatter(xaxis,v_cover,c='green',s=2)
    # plt.show()

    rank=np.zeros((n))

    for v in vlist:

        sc=0
        t=0
        sset=[]
        hmap={}
        for ell in adj_list[v]:

            if(v_cover[ell]>v_cover[v]):
                sc=sc+v_cover[ell]
                t=t+1

            for ell1 in adj_list[ell]:

                if(v_cover[ell1]>v_cover[v]):
                    sc=sc+v_cover[ell1]
                    t=t+1

        if(sc!=0 and t!=0):
            sc=sc/t
        else:
            sc=v_cover[v]


        rank[v]=v_cover[v]/sc

        #rank[v]=sc


    

    v_cover_order=np.zeros((n,2))
    v_cover_order[:,0]=rank
    v_cover_order[:,1]=xaxis


    v_cover_order=sorted(v_cover_order, key=operator.itemgetter(0),reverse=True) 
    v_cover_order=np.array(v_cover_order)

    # print(min(rank),max(rank))

    #Priting the final scores
    # plt.scatter(xaxis,v_cover_order[:,0],c='green',s=2)
    # plt.show()


    return v_cover_order


def FLOW_ng2hopsimple(edge_list,vlist,walk_len_c1,c_const=0):

    n=len(vlist)
    adj_list1=[[] for i in range(n)]
    for (u,v) in edge_list:
        adj_list1[u].append(v)

    adj_list=List(List(x) for x in adj_list1)

    v_cover_order=flow_calc_ng2hopsimple(adj_list,vlist,walk_len_c1,c_const)


    return v_cover_order

    



