import matplotlib.pyplot as plt
import pandas as pd
import metric as met
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import v_measure_score as v_score
from sklearn.metrics import homogeneity_score as homogeneity
from sklearn.metrics import completeness_score as completeness

def metrics_summary(labels,label,res_list,names):
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    n=len(names)
    #print(res_list)
    #flatten res_list into a single list
    res_list_flat = [item for sublist in res_list for item in sublist]
    
    NMI_list=[]
    Purity_list=[]
    V_list=[]
    Homogeneity_list=[]
    Completeness_list=[]
    #Metrics Table
    for i in range(n):
        first = True
        max_NMI = [0,0]
        max_Purity = [0,0]
        max_res = [0,0]
        for k, label_new in enumerate(labels[i]):
            #print('label_new',label_new)
            #print('label_new_len',len(label_new))
            if(round(NMI(label,label_new),3)>max_NMI[0]):
                max_NMI[0] = round(NMI(label,label_new),3)
                max_NMI[1]= round(met.purity_score(label,label_new),3)
                max_res[0] = res_list[i][k]
            if(round(met.purity_score(label,label_new),3)>max_Purity[1]):
                max_Purity[0] = round(NMI(label,label_new),3)
                max_Purity[1]= round(met.purity_score(label,label_new),3)
                max_res[1] = res_list[i][k]
            NMI_list.append(round(NMI(label,label_new),3))
            Purity_list.append(round(met.purity_score(label,label_new),3))
            V_list.append(round(v_score(label,label_new),3))
            Homogeneity_list.append(round(homogeneity(label,label_new),3))
            Completeness_list.append(round(completeness(label,label_new),3))
        print('Max NMI and Purity for',names[i],':\n',max_NMI,'res:',max_res[0],'\n',max_Purity,'res:',max_res[1] )
    
    data = {'NMI':NMI_list,'Purity':Purity_list,'V-score':V_list,'Homogeneity':Homogeneity_list,'Completeness':Completeness_list}
    df = pd.DataFrame(data, index=res_list_flat)
    print(df)

    #NMI vs Purity Plot
    for i in range(n):
        first = True
        for label_new in labels[i]:
            if first:
                first = False
                plt.scatter(round(NMI(label,label_new),3),round(met.purity_score(label,label_new),3),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(round(NMI(label,label_new),3),round(met.purity_score(label,label_new),3),marker='x',color=colors[i])
            
        
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
                plt.scatter(round(completeness(label,label_new),3),round(homogeneity(label,label_new),3),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(round(completeness(label,label_new),3),round(homogeneity(label,label_new),3),marker='x',color=colors[i])
            
        
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
                plt.scatter(res_list[i][idx],round(NMI(label,label_new),3),marker='x',color=colors[i], label = names[i])
            else:
                plt.scatter(res_list[i][idx],round(NMI(label,label_new),3),marker='x',color=colors[i])
            
        
        plt.legend(loc='lower center')
        plt.title('Resolution vs NMI')
        plt.xlabel('Resolution')
        plt.ylabel('NMI') 
    
    