import sys
import os
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import sklearn as skl
import sklearn.linear_model as lm
import sklearn.externals as ex
import sklearn.ensemble as ensemble
import sklearn.metrics as met
import sklearn.model_selection as ms
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=str, nargs='*', help="The filenames or paths to the true positive txts", default=['semmed_tp.txt','mychem_tp.txt','mychem_tp_umls.txt','NDF_TP.txt'])
parser.add_argument("--tn", type=str, nargs='*', help="The filenames or paths to the true negative txts", default=['semmed_tn.txt','mychem_tn.txt','mychem_tn_umls.txt','NDF_TN.txt'])
parser.add_argument("--graph", type=str, help="The filename or path of graph edge file", default="graph_edges.txt")
parser.add_argument("--tncutoff", type=int, help="A positive integer for the true negative cutoff of SemMedDB hot counts to include in analysis", default=2)
parser.add_argument("--tpcutoff", type=int, help="A positive integer for the true positive cutoff of SemMedDB hot counts to include in analysis", default=12)
parser.add_argument("--output", type=str, help="The path of output folder.", default="./")
args = parser.parse_args()

graph_edge = pd.read_csv(args.graph, sep='\t', header=0)
all_nodes = set()
all_nodes.update(set(graph_edge.source))
all_nodes.update(set(graph_edge.target))

TP_list = []
TN_list = []

# generate list of true positive and true negative data frames
for i in range(len(args.tp)):
    temp = pd.read_csv(args.tp[i], sep="\t", index_col=None)
    temp = temp.drop_duplicates().reset_index().drop(columns=['index'])
    select_rows = list(all_nodes.intersection(set(temp['source'])))
    temp = temp.set_index('source').loc[select_rows,:].reset_index()
    select_rows = list(all_nodes.intersection(set(temp['target'])))
    temp = temp.set_index('target').loc[select_rows,:].reset_index()
    if temp.shape[0]!=0:
        TP_list += [temp]
for i in range(len(args.tn)):
    temp = pd.read_csv(args.tn[i], sep="\t", index_col=None)
    temp = temp.drop_duplicates().reset_index().drop(columns=['index'])
    select_rows = list(all_nodes.intersection(set(temp['source'])))
    temp = temp.set_index('source').loc[select_rows,:].reset_index()
    select_rows = list(all_nodes.intersection(set(temp['target'])))
    temp = temp.set_index('target').loc[select_rows,:].reset_index()
    if temp.shape[0]!=0:
        TN_list += [temp]

id_list_class_dict = dict()

# Generate true negative training set by concatinating source-target pair vectors
for TN in TN_list:
    for row in range(len(TN)):
        if 'count' in list(TN):
            if int(TN['count'][row]) < args.tncutoff:
                continue

        source_curie = TN['source'][row]
        target_curie = TN['target'][row]

        if (source_curie, target_curie) not in id_list_class_dict:
            id_list_class_dict[source_curie, target_curie] = 0

# Generate true positive training set by concatinating source-target pair vectors
for TP in TP_list:
    for row in range(len(TP)):
        if 'count' in list(TP):
            if int(TP['count'][row]) < args.tpcutoff:
                continue

        source_curie = TP['source'][row]
        target_curie = TP['target'][row]

        if (source_curie, target_curie) not in id_list_class_dict:
            id_list_class_dict[source_curie, target_curie] = 1
        else:
            if id_list_class_dict[source_curie, target_curie] == 0:
                del id_list_class_dict[source_curie, target_curie]

output_TP = []
output_TN = []
for key, value in id_list_class_dict.items():
    if value == 1:
        output_TP += [key]
    else:
        output_TN += [key]

pd.DataFrame(output_TP, columns=['source', 'target']).to_csv(args.output + "/tp_pairs.txt", sep='\t', index=None)
pd.DataFrame(output_TN, columns=['source', 'target']).to_csv(args.output + "/tn_pairs.txt", sep='\t', index=None)