import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import tensor
from sklearn.metrics import f1_score
plt.switch_backend('agg')

def calculate_f1score(preds, labels, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    f1score = f1_score(labels, preds, average='binary')
    return f1score

def calculate_acc(preds, labels, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = (np.array(preds)>threshold).astype(int)
    acc = (preds == labels).astype(float).mean()
    return acc

def calculate_mrr(drug_disease_pairs, random_pairs):
    '''
    This function is used to calculate Mean Reciprocal Rank (MRR)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    score = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        score += 1/rank
        
    final_score = score/Q_n
    
    return final_score

def calculate_hitk(drug_disease_pairs, random_pairs, k=1):
    '''
    This function is used to calculate Hits@K (H@K)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    count = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        if rank <= k:
            count += 1 
        
    final_score = count/Q_n
    
    return final_score

def plot_cutoff(dfs, plot_title, outfile_path, title_post = ["Random Pairings", "True Negatives", "True Positives"], print_flag=True):

    color = ["xkcd:dark magenta","xkcd:dark turquoise","xkcd:azure","xkcd:purple blue","xkcd:scarlet",
        "xkcd:orchid", "xkcd:pumpkin", "xkcd:gold", "xkcd:peach", "xkcd:neon green", "xkcd:grey blue"]
    c = 0

    for df in dfs:
        cutoffs = [x/100 for x in range(101)]
        cutoff_n = [df["treat_prob"][df["treat_prob"] >= cutoff].count()/len(df) for cutoff in cutoffs]

        plt.plot(cutoffs,cutoff_n,color[c],label=title_post[c])
        if print_flag:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("\n",title_post[c], ":\n")
                print(pd.DataFrame({"cutoff":cutoffs[80:],"count":cutoff_n[80:]}))
        c += 1
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Cutoff Prob')
    plt.ylabel('Rate of Postitive Predictions')
    plt.title(plot_title)
    plt.legend(loc="lower left")
    plt.savefig(outfile_path)
    plt.close()