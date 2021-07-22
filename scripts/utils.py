import pickle
import numpy as np
from datetime import timedelta
from torch.utils.data import Dataset
from torch import tensor
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from sklearn.metrics import f1_score
plt.switch_backend('agg')

def calculate_f1score(preds, labels, average='macro'):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1) 
    labels = np.array(labels)
    f1score = f1_score(labels, y_pred_tags, average=average)
    return f1score

def calculate_acc(preds, labels):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1) 
    labels = np.array(labels)
    acc = (y_pred_tags == labels).astype(float).mean()
    return acc

def calculate_mrr(drug_disease_pairs):
    '''
    This function is used to calculate Mean Reciprocal Rank (MRR)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==0,:].reset_index(drop=True)
    random_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)

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

def calculate_hitk(drug_disease_pairs, k=1):
    '''
    This function is used to calculate Hits@K (H@K)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==0,:].reset_index(drop=True)
    random_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)

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
    
    
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))

class DataWrapper(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        n_id, adjs = pickle.load(open(self.paths[idx],'rb'))
        return (n_id, adjs)

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

def clean_up_desc(string):
    if type(string) is str:
        # Removes all of the "UMLS Semantic Type: UMLS_STY:XXXX;" bits from descriptions
        string = re.sub("UMLS Semantic Type: UMLS_STY:[a-zA-Z][0-9]{3}[;]?", "", string).strip().strip(";")
        if string == 'None':
            return ''
        elif len(re.findall("^COMMENTS: ", string)) != 0:
            return re.sub("^COMMENTS: ","", string)
        elif len(re.findall("-!- FUNCTION: ", string)) != 0:
            part1 = [part for part in string.split('-!-') if len(re.findall("^ FUNCTION: ", part)) != 0][0].replace(' FUNCTION: ','')
            part2 = re.sub(' \{ECO:.*\}.','',re.sub(" \(PubMed:[0-9]*,? ?(PubMed:[0-9]*,?)?\)","",part1))
            return part2
        elif len(re.findall("Check for \"https:\/\/www\.cancer\.gov\/", string)) != 0:
            return re.sub("Check for \"https:\/\/www\.cancer\.gov\/.*\" active clinical trials using this agent. \(\".*NCI Thesaurus\); ","",string)
        else:
            return string
    elif string is None:
        return ''
    else:
        raise ValueError('Not expected type {type(string)}')
        
def clean_up_name(string):
    if type(string) is str:
        if string == 'None':
            return ''
        else:
            return string
    elif string is None:
        return ''
    else:
        raise ValueError('Not expected type {type(string)}')