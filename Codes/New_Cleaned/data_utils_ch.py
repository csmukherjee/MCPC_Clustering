# from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import datasets
import metric as met 
import embedding as embed
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
from collections import Counter


#Set this to whatever you want. Maybe we can write a wrapper here. 
#datapath_main='../'
datapath_main = 'I:/내 드라이브/backup/document/USC/Research/MCPC/final_dataset/Final database/'

def load_data(dataset_name,kchoice=15,pca_dim=50):

    datapath=datapath_main+'data/'

    if dataset_name == 'FashionMNIST':
        import torchvision
        fashion_mnist = torchvision.datasets.FashionMNIST(datapath, download=True)
        num_samples = 30000
        indices = np.random.choice(np.arange(len(fashion_mnist.data)),num_samples)
        X = fashion_mnist.data.flatten(1).numpy()[indices].astype(np.float64)
        label = fashion_mnist.targets.numpy()[indices].astype(int)
        pca = TruncatedSVD(n_components=pca_dim)
        PX = pca.fit_transform(X)   
        print(f'{dataset_name}',PX.shape)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
    elif dataset_name == 'MNIST':
        import torchvision
        mnist = torchvision.datasets.MNIST(datapath, download=True)
        num_samples = 30000
        indices = np.random.choice(np.arange(len(mnist.data)),num_samples)
        X = mnist.data.flatten(1).numpy()[indices].astype(np.float64)
        label = mnist.targets.numpy()[indices].astype(int)
        pca = TruncatedSVD(n_components=pca_dim)
        PX = pca.fit_transform(X)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
        print(f'{dataset_name}',PX.shape)
    elif dataset_name == 'seeds':
        dataset_path = datapath+'seeds_dataset.csv'
        X = pd.read_csv(dataset_path)
        label = X.iloc[:,-1:].values[:,0] - 1
        X = X.iloc[:,:-1].values
        X=(X-X.mean())/X.std()
        print(f'{dataset_name}',X.shape)
        edge_list,vlist=embed.dir_KNN_graph(X,kchoice,0)
    elif dataset_name == 'breast-cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        label = data.target
        X = data.data
        X=(X-X.mean())/X.std()
        edge_list,vlist=embed.dir_KNN_graph(X,kchoice,0)
    elif dataset_name == 'Omniglot':
        import torchvision
        omniglot = torchvision.datasets.Omniglot(datapath, download=True, transform=torchvision.transforms.ToTensor())
        num_samples = 10000
        indices = np.random.choice(np.arange(len(omniglot)),num_samples)
        X = []
        label = []
        for ind in indices:
            img, lab = omniglot[ind]
            X.append(img.flatten(1).numpy())
            label.append(lab)
        X = np.concatenate(X).astype(np.float64)
        label = np.array(label).astype(int)
        pca = TruncatedSVD(n_components=pca_dim)
        PX = pca.fit_transform(X)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
    elif dataset_name == 'KMNIST':
        import torchvision
        kmnist = torchvision.datasets.KMNIST(datapath, download=True, transform=torchvision.transforms.ToTensor())
        num_samples = 30000
        indices = np.random.choice(np.arange(len(kmnist)),num_samples)
        X = []
        label = []
        for ind in indices:
            img, lab = kmnist[ind]
            X.append(img.flatten(1).numpy())
            label.append(lab)
        X = np.concatenate(X).astype(np.float64)
        label = np.array(label).astype(int)
        pca = TruncatedSVD(n_components=pca_dim)
        PX = pca.fit_transform(X)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
    elif dataset_name == 'shuttle':
        dataset_path = datapath+'statlog+shuttle/'
        df_train = pd.read_csv(dataset_path+"shuttle.trn", delimiter=' ', header=None)
        df_test = pd.read_csv(dataset_path+"shuttle.tst", delimiter=' ', header=None)
        df_train = df_train[df_train.iloc[:, -1] != 6]
        df_train = df_train[df_train.iloc[:, -1] != 7]
        df_train = df_train[df_train.iloc[:, -1] != 2]
        # df_train = df_train[df_train.iloc[:, -1] != 3]
        df_test = df_test[df_test.iloc[:, -1] != 6]
        df_test = df_test[df_test.iloc[:, -1] != 7]
        df_test = df_test[df_test.iloc[:, -1] != 2]
        # df_test = df_test[df_test.iloc[:, -1] != 3]

        X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
        X = pd.concat([X_train, X_test])
        label = pd.concat([y_train, y_test])
        print(label.count())
        X=(X-X.mean())/X.std()
        print(f'{dataset_name}',X.shape)
        edge_list,vlist=embed.dir_KNN_graph(X,kchoice,0)


    elif dataset_name in ['bbc_news', 'BBC_Sports', 'biorxiv', 'reddit', '20NewsGroups', 'big_patent']:
        features = torch.load(datapath+f'{dataset_name}/features.pt')
        label = torch.load(datapath+f'{dataset_name}/labels.pt')
        pca = TruncatedSVD(n_components=pca_dim)
        PX = pca.fit_transform(features)
        print(f'{dataset_name}',PX.shape)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
    
    elif dataset_name == '20NewsGroups_tfdif':
        import datasets
        from sklearn.feature_extraction.text import TfidfVectorizer
        raw_data = datasets.load_dataset("mteb/twentynewsgroups-clustering")['test'][9]
        label = np.array(raw_data['labels'])
        text = raw_data['sentences']
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

        features = tfidf.fit_transform(text).toarray() # Remaps the words in the 1490 articles in the text column of 
                                                          # data frame into features (superset of words) with an importance assigned 
                                                          # based on each words frequency in the document and across documents
        pca = TruncatedSVD(n_components=50)
        PX = pca.fit_transform(features)
        print(f'{dataset_name}',PX.shape)
        edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)

    else:
        print("Name of dataset entered is incorrect")

    # else:
    #     # scRNA_datapath = '../data/pca-benchmarks/'
    #     scRNA_datapath = datapath_main+'scRNA_benchmark/'
    #     X = scipy.sparse.load_npz(scRNA_datapath+dataset_name + '/data.npz')
    #     label = np.load(scRNA_datapath+dataset_name+'/labels.npy')
    #     X.data = np.log1p(X.data)
    #     print("Log transform done")
    #     pca = TruncatedSVD(n_components=20)
    #     PX = pca.fit_transform(X)
    #     n=PX.shape[0]
    #     walk_len_c1=int(np.log2(n))
    #     print(PX.shape)
    #     #Calculte inital KNN accuracy
    #     met.KNN_graph_acc(PX,kchoice,0,label)
    #     edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)

    return edge_list, label

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def preprocess_bbc_sport():
    datapath = datapath+'BBC_Sports/'
    text = []
    label = []
    clusters = ['athletics', 'cricket', 'football', 'rugby', 'tennis']
    for cluster_id, cluster in enumerate(clusters):
        dir_list = os.listdir(datapath+cluster+'/')
        for file in dir_list:
            with open(datapath+cluster+'/'+file, 'r') as f: 
                text.append(f.read())
                label.append(cluster_id)
    df = pd.DataFrame(columns = ['text', 'label'])
    df['text'] = text
    df['label'] = label
    df.to_csv(datapath+'/data.csv')

def save_embedding(ds_name = 'biorxiv'):
    from tqdm import tqdm
    datapath = datapath_main+'data/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ds_name == 'biorxiv':
        import datasets
        raw_data = datasets.load_dataset("mteb/raw_biorxiv")["train"]
        label = raw_data['category']
        text = [m+n for m,n in zip(raw_data['title'],raw_data['abstract'])]
        cluster_idx, count = np.unique(label, return_counts=True)
    elif ds_name == 'reddit':
        import datasets
        raw = datasets.load_dataset("mteb/reddit-clustering")['test'][2]
        label_str = raw['labels']
        label_name, count = np.unique(label_str,return_counts=True)
        label_to_id_map = {}
        for cluster_id, name in enumerate(label_name):
            label_to_id_map[name] = cluster_id
        label = np.array([label_to_id_map[x] for x in label_str])
        text = raw['sentences']
    elif ds_name == 'BBC_Sports':
        dataset_path = datapath+'BBC_Sports/data.csv'
        X = pd.read_csv(dataset_path)
        label = X['label']
        text = X['text']
    elif ds_name == '20NewsGroups':
        import datasets
        from sklearn.feature_extraction.text import TfidfVectorizer
        raw_data = datasets.load_dataset("mteb/twentynewsgroups-clustering")['test'][9]
        label = np.array(raw_data['labels'])
        text = raw_data['sentences']
    elif ds_name == 'bbc_news':
        dataset_path = datapath+'bbc_news_train.csv'
        X = pd.read_csv(dataset_path)
        X['category_id'] = X['Category'].factorize()[0]
        label = X.category_id 
        text = list(X.Text)
    elif ds_name == 'big_patent':
        import datasets
        raw = datasets.load_dataset("jinaai/big-patent-clustering")['test'][2]
        label_str = raw['labels']
        label_name, count = np.unique(label_str,return_counts=True)
        label_to_id_map = {}
        for cluster_id, name in enumerate(label_name):
            label_to_id_map[name] = cluster_id
        label = np.array([label_to_id_map[x] for x in label_str])
        text = raw['sentences']

    model_path = 'Alibaba-NLP/gte-base-en-v1.5'
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch_list = list(chunks(text, 128))
    features = []
    for batch in tqdm(batch_list):
        batch_dict = tokenizer(batch, max_length=256, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(
                batch_dict['input_ids'].to(device), 
                attention_mask = batch_dict['attention_mask'].to(device)
            )
        embeddings = outputs.last_hidden_state[:, 0]
        features.append(F.normalize(embeddings, p=2, dim=1))
    features = torch.cat(features)
    features = features.cpu().detach().numpy()
    print(features.shape)
    os.makedirs(datapath+f'{ds_name}', exist_ok=True)
    torch.save(features, datapath+f'{ds_name}/features.pt')
    torch.save(np.array(label), datapath+f'{ds_name}/labels.pt')


#scRNA data
def local_SCRNA(name,kchoice=15):

    datapath=datapath_main+'SCRNA_benchmark/'

    X = scipy.sparse.load_npz(datapath+name + '/data.npz')
    label = np.load(datapath+name+'/labels.npy')
    print(name,len(label))

    #Log transform+PCA
    X.data = np.log1p(X.data)
    print("Log transform done")
    pca = TruncatedSVD(n_components=50)
    PX = pca.fit_transform(X)
    n=PX.shape[0]
    walk_len_c1=int(np.log2(n))
    print(PX.shape)


    #Calculte inital KNN accuracy
    met.KNN_graph_acc(PX,kchoice,0,label)

    #Get the KNN edgelist
    edge_list,vlist=embed.dir_KNN_graph(PX,kchoice,0)
    print(len(edge_list))

    return edge_list,vlist,n,label

import networkx as nx

#Graph data
def graph_database(dataname,good_v=0):    


    #Cora graph
    if(dataname=='Cora'):

        import csv
        edge_list00=[]
        cora_path=datapath_main+'graph-data/cora-edges.csv'
        with open(cora_path, mode ='r')as file:
            csvFile = csv.reader(file)
            c=0
            for lines in csvFile:
                if(c>0):
                    edge_list00.append((int(lines[1]),int(lines[2])))
                    
                c=c+1


        G=nx.DiGraph(edge_list00)
        sets=[x for x in nx.weakly_connected_components(G)]
        zlist=[]
        for x in sets:
            if(len(x)>26):
                for y in x:
                    zlist.append(y)

        edge_list0=[]

        for(u,v) in edge_list00:
            if(u in zlist and v in zlist):
                edge_list0.append((u,v))


        hmap={}


        t=0
        for (u,v) in edge_list0:

            if(u in hmap):
                continue
            else:
                hmap[u]=t
                t=t+1

        n=t
        edge_list=[]
        for (u,v) in edge_list0:
            edge_list.append((hmap[u],hmap[v]))

        label0=[''] * n
        cora_node_path=datapath_main+'graph-data/cora-nodes.csv'
        with open(cora_node_path, mode ='r')as file:
            csvFile = csv.reader(file)
            c=0
            for lines in csvFile:
                if(c==0):
                    c=c+1
                    continue


                else:
                    x=int(lines[1])
                    y=lines[3]
                    if(x in zlist):
                        label0[hmap[x]]=y
                        c=c+1


        hmap1={}
        tt=0      
        for x in label0:
            if(x in hmap1):
                continue
            
            else:
                hmap1[x]=tt
                tt=tt+1

        label=[]
        for x in label0:
            label.append(hmap1[x])

        vlist=[i for i in range(n)]


    #Cora full
    if(dataname=='Cora full'):
        #cora full

        edge_list00=[]
        fpath=datapath_main+'graph-data/cora.edges'
        flabel=datapath_main+'graph-data/cora.clusters'

        edge_list=[]

        for line in open(fpath):
            u, v = map(int, line.rstrip().split("\t"))
            edge_list.append((u-1,v-1))


        label=[]
        with open(flabel) as f:
            t=0
            for line in f:
                    l1=line.rstrip()
                    l2=[int(x) for x in l1.split("\t")]
                    label.append(l2[1])

        n=len(label)
        vlist=[i for i in range(n)]

        in_degs=np.zeros((n))
        out_degs=np.zeros((n))
        for (u,v) in edge_list:
            out_degs[u]+=1
            in_degs[v]+=1


        # good_v=[]
        # hmap_v={}
        # t=0
        # for u in range(n):
        #     if(out_degs[u]==0):
        #         edge_list.append((u,u))
            
        #     else:
        #         good_v.append(u)
        #         hmap_v[u]=t
        #         t=t+1

    #Citeseer
    if(dataname=='Citeseer'):
        data_loc=datapath_main+'graph-data/citeseer-doc-classification/'
        graph_file = open(data_loc+'citeseer.cites', 'r')
        graph_file.seek(0)
        iid = {}  # Integer id conversion dict
        idx = 0
        citeseer_edgelist = []
        for line in graph_file.readlines():
            i, j = line.split()
            if i not in iid:
                iid[i] = idx
                idx += 1
            if j not in iid:
                iid[j] = idx
                idx += 1
            citeseer_edgelist.append((iid[j],iid[i]))


        graph_file.close()

        citeseer = nx.DiGraph(citeseer_edgelist)

        # Prepare data arrays and labels lookup table
        citeseer_labels = np.ndarray(shape=(len(iid)), dtype=int)
        citeseer_features = np.ndarray(shape=(len(iid), 3703), dtype=int)
        labels = {'Agents': 0, 'AI': 1, 'DB': 2, 'IR': 3, 'ML': 4, 'HCI': 5}
        no_labels = set(citeseer.nodes())

        # Read data
        with open(data_loc+'citeseer.content', 'r') as f:
            for line in f.readlines():
                oid, *data, label = line.split()
                citeseer_labels[iid[oid]] = labels[label]
                citeseer_features[iid[oid],:] = list(map(int, data))
                no_labels.remove(iid[oid])
                
        for i in no_labels:
            citeseer_labels[i] = -1
            citeseer_features[i,:] = np.zeros(3703)
            
        # Validation
        with open(data_loc+'citeseer.content', 'r') as f:
            for line in f.readlines():
                oid, *data, label = line.split()
                assert citeseer_labels[iid[oid]] == labels[label]
                assert citeseer_labels[iid[oid]] < 6
                assert sum(citeseer_features[iid[oid]]) == sum(map(int, data))
            print("Validation for `citeseer_labels` and `citeseer_features` passes.")



        #If we do not remove anything
        # edge_list=citeseer_edgelist.copy()
        # label=citeseer_labels.copy()
        # n=len(label)
        # vlist=[i for i in range(n)]

        #Otherwise
        n=len(citeseer_labels)
        sets=[x for x in nx.weakly_connected_components(citeseer)]
        new_list=sets[0]
        hmap={}
        t=0
        for i in new_list:
            hmap[i]=int(t)
            t=t+1

        edge_list=[]
        for (u,v) in citeseer_edgelist:
            if(u in new_list and v in new_list):
                edge_list.append((hmap[u],hmap[v]))

        label=citeseer_labels[list(new_list)].copy()
        n=len(label)
        vlist=[i for i in range(n)]

        degs=np.zeros((n))
        indegs=np.zeros((n))
        for (u,v) in edge_list:
            degs[u]=degs[u]+1
            indegs[v]=indegs[v]+1


        # x=0
        # good_v=[]
        # for i in range(n):
        #     if(degs[i]==0):
        #         edge_list.append((i,i))
        #         x=x+1
        #     else:
        #         good_v.append(i)

        #print(x)


        
    #Eu core
    if(dataname=='Eu core'):
        edge_list=[]
        eu_graph=datapath_main+'graph-data/email-Eu-core.txt'
        eu_label=datapath_main+'graph-data/email-Eu-core-labels.txt'

        with open(eu_graph) as f:
            for line in f:
                    l1=line.rstrip()
                    l2=[int(x) for x in l1.split()]
                    edge_list.append((l2[0],l2[1]))

        f.close()
        label=[]
        with open(eu_label) as f:
            t=0
            for line in f:
                    l1=line.rstrip()
                    l2=[int(x) for x in l1.split()]
                    label.append(l2[1])

        n=len(label)
        vlist=[i for i in range(n)]

        in_degs=np.zeros((n))
        out_degs=np.zeros((n))
        for (u,v) in edge_list:
            out_degs[u]+=1
            in_degs[v]+=1
            


    
    # if(good_v==1):

    #     out_degs=np.zeros((n))
    #     for (u,v) in edge_list:
    #         out_degs[u]+=1
        
    #     c=0
    #     good_v=[]
    #     for u in range(n):
    #         if(out_degs[u]==0):
    #             c=c+1
    #         else:
    #             good_v.append(u)
        
    #     print("Graph name:=",dataname)
    #     print("|V|=",n,", |E|=",len(edge_list),",cluster_num=",len(set(label))," and #sink vertices",c)

    #     for u in range(n):
    #         if(out_degs[u]==0):
    #                 edge_list.append((u,u)) 


    return edge_list,vlist,label,n,good_v
        

#Bulk-RNA data
def process(X,labels):

   
    lset=labels[:,0]


    idx0=[]
    for i in range(X.shape[0]):
        if(X[i,0] in lset):
            idx0.append(i)
    idx0=np.array(list(idx0))
    print(idx0[0:10])

    X1=X[idx0,:].copy()
    print(X1.shape)


     
    idx=[]
    for i in range(X1.shape[0]):
        c=0
        t1=X1[i,0]
        for j in lset:
            if(j==t1):
                idx.append(c)
                break
            c=c+1
    print('selected vertices ',len(idx))

    label=labels[idx,1]
    print(label.shape,Counter(label))


    Y=np.log2(X1[:,1:].astype(float)+1)

    edge_list,vlist=embed.dir_KNN_graph(Y,15,100)


    return edge_list,vlist,label


def local_bulkRNA(name,survive=0):
    datapath1=datapath_main+'Multiomics/'


    if(name=='miRNA'):
        df_labels1 = pd.read_csv(datapath1+'sample_sheet_mirna.csv')
        x3=df_labels1[['Sample ID','Project ID']]
        dfl=x3.to_numpy()
        dfr=pd.read_csv(datapath1+'miRNA_raw.csv')
        Xdf=dfr.to_numpy()



    
    if(name=='mRNA'):
        df_labels2 = pd.read_csv(datapath1+'sample_sheet_mrna.csv')
        x3=df_labels2[['Sample ID','Project ID']]
        dfl=x3.to_numpy()
        dfr=pd.read_csv(datapath1+'mRNA_pc_gene_raw.csv')
        Xdf=dfr.to_numpy()

    if(survive==1):
        survive=pd.read_csv(datapath1+'survival.csv')
        survive=survive.to_numpy()
        dfsurvive=survive[:,[0,2]]

        Xdft=Xdf.copy()
        for i in range(Xdft.shape[0]):
            ell=Xdft[i,0]
            ell1=ell[:-4]
            Xdft[i,0]=ell1

        Xdf=Xdft.copy()
        dfl=dfsurvive.copy()


    edge_list,vlist,label=process(Xdf,dfl)
    n=len(label)
    vlist=[i for i in range(n)]


    
    return edge_list,vlist,label,n



#Set labels.
def set_labels(label):

    label_new=[]
    hmap={}
    t=0
    for i in label:
        if i not in hmap:
            hmap[i]=t
            t=t+1

    for i in label:
        label_new.append(hmap[i])

    label=label_new.copy()
    print(len(set(label)),max(label))

    return label