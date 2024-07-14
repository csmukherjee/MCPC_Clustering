import matplotlib.pyplot as plt
import pandas as pd
import metric as met
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import v_measure_score as v_score
from sklearn.metrics import homogeneity_score as homogeneity
from sklearn.metrics import completeness_score as completeness

def metrics_summary(labels,label,res_list,names):
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    n=len(labels)

    
    
    NMI_list=[]
    Purity_list=[]
    V_list=[]
    Homogeneity_list=[]
    Completeness_list=[]
    #Metrics Table
    for i in range(n):
        first = True
        for label_new in labels[i]:
            NMI_list.append(round(NMI(label,label_new),2))
            Purity_list.append(round(met.purity_score(label,label_new),2))
            V_list.append(round(v_score(label,label_new),2))
            Homogeneity_list.append(round(homogeneity(label,label_new),2))
            Completeness_list.append(round(completeness(label,label_new),2))
    data = {'NMI':NMI_list,'Purity':Purity_list,'V-score':V_list,'Homogeneity':Homogeneity_list,'Completeness':Completeness_list}
    df = pd.DataFrame(data, index=res_list*len(names))
    print(df)

    #NMI vs Purity Plot
    for i in range(n):
        first = True
        for label_new in labels[i]:
            NMI_list.append(round(NMI(label,label_new),2))
            Purity_list.append(round(met.purity_score(label,label_new),2))
            V_list.append(round(v_score(label,label_new),2))
            Homogeneity_list.append(round(homogeneity(label,label_new),2))
            Completeness_list.append(round(completeness(label,label_new),2))
            if first:
                first = False
                plt.scatter(round(NMI(label,label_new),2),round(met.purity_score(label,label_new),2),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(round(NMI(label,label_new),2),round(met.purity_score(label,label_new),2),marker='x',color=colors[i])
            
        
        plt.legend(loc='lower center')
        plt.title('NMI vs Purity')
        plt.xlabel('NMI')
        plt.ylabel('Purity')
        
    plt.show()

    #Completeness vs Homogeneity Plot
    for i in range(n):
        first = True
        for label_new in labels[i]:
            if first:
                first = False
                plt.scatter(round(completeness(label,label_new),2),round(homogeneity(label,label_new),2),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(round(completeness(label,label_new),2),round(homogeneity(label,label_new),2),marker='x',color=colors[i])
            
        
        plt.legend(loc='lower center')
        plt.title('Completeness vs Homogeneity')
        plt.xlabel('Completeness')
        plt.ylabel('Homogeneity')
    plt.show() 

    #resolution vs NMI
    for i in range(n):
        first = True
        for idx, label_new in enumerate(labels[i]):
            if first:
                first = False
                plt.scatter(res_list[idx],round(NMI(label,label_new),2),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(res_list[idx],round(NMI(label,label_new),2),marker='x',color=colors[i])
            
        
        plt.legend(loc='lower center')
        plt.title('Resolution vs NMI')
        plt.xlabel('Resolution')
        plt.ylabel('NMI') 
    
    