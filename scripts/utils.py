import pickle
from datetime import timedelta
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import re
plt.switch_backend('agg')

def calculate_acc(preds, labels, threshold=0.5):
    preds = (preds>=threshold).float()
    acc = (preds == labels).float().mean()
    return acc

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

def plot_cutoff(dfs, outloc, title_post = ["Random Pairings", "True Negatives", "True Positives"], print_flag=True):

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
    plt.title('Prediction Rates of Treats Class')
    plt.legend(loc="lower left")
    outloc = outloc + '/Figure1.png'
    plt.savefig(outloc)
    plt.close()

def clean_up_desc(string):
    if type(string) is str:
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