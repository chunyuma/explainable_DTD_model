import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sklearn.externals as ex
import sklearn.ensemble as ensemble
import sklearn.metrics as met
from sklearn.linear_model import LogisticRegression

import time
import argparse
import joblib
from glob import glob
from utils import calculate_acc, plot_cutoff, calculate_f1score, calculate_mrr, calculate_hitk

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

def generate_X_and_y(data_df, curie_to_vec_dict, pair_emb='concatenate'):
    
    if pair_emb == 'concatenate':
        X = np.vstack([np.hstack([curie_to_vec_dict[data_df.loc[index,'source']], curie_to_vec_dict[data_df.loc[index,'target']]]) for index in range(len(data_df))])
        y = np.array(list(data_df['y']))
        
    elif pair_emb == 'hadamard':
        X = np.vstack([curie_to_vec_dict[data_df.loc[index,'source']]*curie_to_vec_dict[data_df.loc[index,'target']] for index in range(len(data_df))])
        y = np.array(list(data_df['y']))
    
    else:
        raise TypeError("Only 'concatenate' or 'hadamard' is acceptable")
        
    return [X, y]

def generate_X_and_y_from_txt(data_dir, curie_to_vec_dict):
    with open(data_dir, 'r') as f:
        data = f.readlines()
    data = [l.strip('\n').split('\t') for l in data[1:]]
    X = np.zeros([len(data), 1024])
    y = np.zeros([len(data)])
    for i, item in enumerate(data):
        X[i] = np.hstack([curie_to_vec_dict[item[0]], curie_to_vec_dict[item[1]]])
        y[i] = int(item[2])

    return [X, y]


def evaluate(model, X=None, y_true=None, loader=None, num_sample=0, calculate_metric=True): 

    if loader:
        model.eval()
        probas = np.zeros([num_sample,3])
        y_true = np.zeros([num_sample])
        with torch.no_grad():
            last_end = 0
            for X, y in loader:
                l = len(y)
                X, y = X.to(device), y.to(device)
                output = model(X)
                probas[last_end:last_end+l] = output.cpu().numpy()
                y_true[last_end:last_end+l] = y.cpu().numpy()
                last_end += l
    else:
        probas = model.predict_proba(X)
    
    if calculate_metric is True:
        
        ## calculate accuracy
        acc = calculate_acc(probas, y_true)
        ## calculate F1 score
        f1score = calculate_f1score(probas,y_true)
        ## calculate AUC
        auc_score = met.roc_auc_score(y_true, probas, multi_class='ovr')

        ## calculate AP (average precision) score
        ## generate Receiver operating characteristic (ROC) curve plot data
        ## generate Precision Recall Curve plot data
        if probas.shape[1] <= 2:
            ap_score = met.average_precision_score(y_true, probas)
            fpr, tpr, _ = met.roc_curve(y_true, probas)
            precision, recall, _ = met.precision_recall_curve(y_true, probas)
        else:
            ap_score = 0
            fpr = 0
            tpr = 0
            precision = 0
            recall = 0
        plot_data = dict()
        plot_data['roc_curve'] = {'fpr':fpr, 'tpr':tpr}
        plot_data['precision_recall_curve'] = {'precision':precision, 'recall':recall}
        

        
        # ## generate detection error tradeoff (DET) curve plot data
        # fpr, fnr, _ = met.det_curve(y_true, probas)
        # plot_data['det_curve'] = {'fpr':fpr, 'fnr':fnr}
        
        return [acc, f1score, auc_score, ap_score, plot_data, y_true, probas]
    
    else:
        
        return [None, None, None, None, None, y_true, probas]



def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NodeDataset(Dataset):
    def __init__(self, vecs, labels):
        self.vecs = vecs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.vecs[index], self.labels[index]


class Net(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, 2)
        # self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3 = nn.Linear(hidden_size, 3)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.Softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        x = x.float()
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        output = x
        output = self.Softmax(x)
        return output


def train(model, device, train_loader, optimizer, scheduler, epoch, log_step):
    model.train()
    for batch_idx, (sequence, label) in enumerate(train_loader):
        sequence, label = sequence.to("cuda"), label.to("cuda")
        optimizer.zero_grad()
        output = model(sequence)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, label.long())
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(output, label.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        if batch_idx % log_step == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sequence), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_data_from_txt", type=str, help="Whether to load data from txt", default='')
    parser.add_argument("--run_mode", type=int, help="Model for running model. 1 for normal mode, 2 for crossvalidation", default=1)
    parser.add_argument("--data_path", type=str, help="Data Forlder", default='~/work/GraphSage_RF_model/data')
    parser.add_argument("--depth", type=int, help="a positive integer for the maximum depth of trees in the random forest", default=15)
    parser.add_argument("--trees", type=int, help="a positive integer for the number of trees in the random forest", default=2000)
    parser.add_argument("--m_suffix", type=str, help="Model Name Suffix", default='default')
    parser.add_argument("--pair_emb", type=str, help="The method for the pair embedding (concatenate or hadamard).", default="concatenate")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/work/GraphSage_RF_model/results")
    parser.add_argument("--model", type=str, help="Choose from RF, NN and LR", default='RF')
    args = parser.parse_args()
    start_time = time.time()


    print("Loading Graphsage Embeddings")
    with open(args.data_path + "/all_node_vec.pkl", 'rb') as infile:
        curie_to_vec_dict = pickle.load(infile)
    
    if args.run_mode == 1:
        ## read train set, validation set and test set (ignore the validation set)
        print('Start processing data', flush=True)

        if args.load_data_from_txt != '':
            train_X, train_y = generate_X_and_y_from_txt(os.path.join(args.load_data_from_txt, 'train_pairs.txt'), curie_to_vec_dict)
            val_X, val_y = generate_X_and_y_from_txt(os.path.join(args.load_data_from_txt, 'val_pairs.txt'), curie_to_vec_dict)
            test_X, test_y = generate_X_and_y_from_txt(os.path.join(args.load_data_from_txt, 'test_pairs.txt'), curie_to_vec_dict)
            train_data = pd.read_csv(os.path.join(args.load_data_from_txt, 'train_pairs.txt'), sep='\t', header=0)
            val_data = pd.read_csv(os.path.join(args.load_data_from_txt, 'val_pairs.txt'), sep='\t', header=0)
            test_data = pd.read_csv(os.path.join(args.load_data_from_txt, 'test_pairs.txt'), sep='\t', header=0)

        else:
            with open(args.data_path + '/mode1/train_val_test.pkl', 'rb') as infile:
                train_batch, val_batch, test_batch = pickle.load(infile)
                train_data = pd.concat(train_batch).reset_index(drop=True)
                val_data = pd.concat(val_batch).reset_index(drop=True)
                test_data = pd.concat(test_batch).reset_index(drop=True)

            train_X, train_y = generate_X_and_y(train_data, curie_to_vec_dict, pair_emb='concatenate')
            val_X, val_y = generate_X_and_y(val_data, curie_to_vec_dict, pair_emb='concatenate')
            test_X, test_y = generate_X_and_y(test_data, curie_to_vec_dict, pair_emb='concatenate')

        print(train_X.shape)
        print(val_X.shape)
        print(test_X.shape)



        folder_name = f'RF_trees{args.trees}_maxdepth{args.depth}_{args.m_suffix}'
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name))
        except:
            pass
        
        # Sets and fits Random ForestModel
        print('Start training model', flush=True)
        if args.model == 'NN':
            lr = 0.0005
            batch_size = 32
            num_epoch = 100
            hidden_size = 2048
            dropout = 0.1
            log_step = 100
            device = torch.device('cuda')
            gamma = 0.95
            seed = 24

            set_seed(seed)

            print('Process data for neural network model', flush=True)
            train_dataset = NodeDataset(vecs=train_X, labels=train_y)
            val_dataset = NodeDataset(vecs=val_X, labels=val_y)
            test_dataset = NodeDataset(vecs=test_X, labels=test_y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print('Start training neural network model', flush=True)
            fitModel = Net(hidden_size=hidden_size, dropout=dropout).to(device)
            optimizer = optim.AdamW(fitModel.parameters(), lr=lr)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            for epoch in range(1, num_epoch + 1):
                print('Start epoch: {}'.format(epoch), flush=True)
                train(fitModel, device, train_loader, optimizer, scheduler, epoch, log_step)


            print("")
            print('#### Evaluate best model ####', flush=True)
            # train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(fitModel, loader=train_loader, num_sample=len(train_dataset))
            train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(fitModel, loader=train_loader, num_sample=len(train_dataset))
            val_acc, val_f1score, val_auc_score, val_ap_score, val_plot_data, val_y_true, val_y_probs = evaluate(fitModel, loader=val_loader, num_sample=len(val_dataset))
            test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(fitModel, loader=test_loader, num_sample=len(test_dataset))
        else:
            if args.model == 'RF':
                RF_model = ensemble.RandomForestClassifier(class_weight='balanced', max_depth=args.depth, max_leaf_nodes=None, n_estimators=args.trees, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", n_jobs=-1)
                fitModel = RF_model.fit(train_X, train_y)
            elif args.model == 'LR':
                logistic_model = LogisticRegression(class_weight='balanced', penalty='elasticnet', solver='saga', l1_ratio=0.5)
                fitModel = logistic_model.fit(train_X, train_y)
            else:
                raise ValueError("Choose model from RF, NN and LR")
            print("")
            print('#### Evaluate best model ####', flush=True)
            train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(fitModel, train_X, train_y)
            val_acc, val_f1score, val_auc_score, val_ap_score, val_plot_data, val_y_true, val_y_probs = evaluate(fitModel, val_X, val_y)
            test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)
            



        test_data['prob'] = test_y_probs[:,0]
        # _, _, _, _, _, random_y_true, random_y_probs = evaluate(fitModel, random_X, random_y, False)
        # random_data['prob'] = random_y_probs
        
        test_mrr_score = calculate_mrr(test_data,)
        test_hit1_score = calculate_hitk(test_data, k=1)
        test_hit10_score = calculate_hitk(test_data, k=10)
        test_hit20_score = calculate_hitk(test_data, k=20)
        test_hit50_score = calculate_hitk(test_data, k=50)     
        print(f'Final AUC: Train Auc: {train_auc_score:.5f}, Test Auc: {val_auc_score:.5f}, Test Auc: {test_auc_score:.5f}')
        print(f'Final Accuracy: Train Accuracy: {train_acc:.5f}, Test Accuracy: {val_acc:.5f}, Test Accuracy: {test_acc:.5f}')
        print(f'Final F1 score: Train F1score: {train_f1score:.5f}, Test F1score: {val_f1score:.5f}, Test F1score: {test_f1score:.5f}')
        print(f'Final AP score: Train APscore: {train_ap_score:.5f}, Test APscore: {val_ap_score:.5f}, Test APscore: {test_ap_score:.5f}')
        print(f"MRR score for test data: {test_mrr_score:.5f}")
        print(f"Hit@1 for test data: {test_hit1_score:.5f}")
        print(f"Hit@10 for test data: {test_hit10_score:.5f}")
        print(f"Hit@20 for test data: {test_hit20_score:.5f}")
        print(f"Hit@50 for test data: {test_hit50_score:.5f}")
        
        ## Saves all evaluation result data for downstream analysis
        all_evaluation_results = dict()
        all_evaluation_results['evaluation_acc_score'] = [train_acc, val_acc, test_acc]
        all_evaluation_results['evaluation_f1_score'] = [train_f1score, val_f1score, test_f1score]
        all_evaluation_results['evaluation_auc_score'] = [train_auc_score, val_auc_score, test_auc_score]
        all_evaluation_results['evaluation_ap_score'] = [train_ap_score, val_ap_score, test_ap_score]
        all_evaluation_results['evaluation_plot_data'] = [train_plot_data, val_plot_data, test_plot_data]
        # all_evaluation_results['evaluation_y_true'] = [train_y_true, val_y_true, (test_y_true, random_y_true)]
        # all_evaluation_results['evaluation_y_probas'] = [train_y_probs, val_y_probs, (test_y_probs, random_y_probs)]
        with open(os.path.join(args.output_folder, folder_name, 'all_evaluation_results.pkl'),'wb') as file:
            pickle.dump(all_evaluation_results, file)
        
        ## Plots Receiver operating characteristic (ROC) curve
        print("plot Receiver operating characteristic (ROC) curve", flush=True)
        for datatype in ['train', 'val', 'test']:
            temp_plot_data = eval(datatype+'_plot_data')
            fpr, tpr = temp_plot_data['roc_curve']['fpr'], temp_plot_data['roc_curve']['tpr']
            auc_score = eval(datatype+'_auc_score')
            f1_score = eval(datatype+'_f1score')
            plt.plot(fpr, tpr, lw=2, label=f'{datatype.capitalize()} Data (AUC = {auc_score:.5f}, F1score = {f1_score:.5f})')

        # Plots the 50/50 line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Coin Flip', alpha=.8)

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, 'roc_vurve.png')
        plt.savefig(outloc)
        plt.close()

        ## Plots Precision Recall (PR) curve
        print("plot Precision Recall (PR) curve", flush=True)
        for datatype in ['train', 'val', 'test']:
            temp_plot_data = eval(datatype+'_plot_data')
            precision, recall = temp_plot_data['precision_recall_curve']['precision'], temp_plot_data['precision_recall_curve']['recall']
            ap_score = eval(datatype+'_ap_score')
            plt.plot(recall, precision, lw=2, label=f'{datatype.capitalize()} Data (AP = {ap_score:.5f})')

        # Plots the 50/50 line
        plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='black', label='Coin Flip', alpha=.8)

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall (PR) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, 'precision-recall_curve.png')
        plt.savefig(outloc)
        plt.close()

        # ## Plots Detection Error Tradeoff (DET) curve
        # print("plot Detection Error Tradeoff (DET) curve", flush=True)
        # for datatype in ['train', 'val', 'test']:
        #     temp_plot_data = eval(datatype+'_plot_data')
        #     fpr, fnr = temp_plot_data['det_curve']['fpr'], temp_plot_data['det_curve']['fnr']
        #     plt.plot(fpr, fnr, lw=2, label=f'{datatype.capitalize()} Data')

        # # Sets legend, limits, labels, and displays plot
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('False Negative Rate')
        # plt.title('Detection Error Tradeoff (DET) Curve')
        # plt.legend(loc="lower right")
        # plt.tight_layout()
        # outloc = os.path.join(args.output_folder, folder_name, 'det_curve.png')
        # plt.savefig(outloc)
        # plt.close()

        
        ## Plots Rate Cutoff for test dataset
        probas_tp = test_y_probs[np.where(test_y_true == 1)[0]]
        probas_tn = test_y_probs[np.where(test_y_true == 0)[0]]
        # plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
        # pd.DataFrame({"treat_prob":[pr for pr in probas_tn]}),
        # pd.DataFrame({'treat_prob':[pr for pr in random_y_probs]})],
        # 'Prediction Rates of Treats Class for Test dataset',
        # os.path.join(args.output_folder, folder_name, 'test_dataset_rate_cutoff.png'),
        # ["True Positives", "True Negatives", "Random Pairs"])
        
    elif args.run_mode == 2:
        
        fold_list = sorted(glob(args.data_path +'/mode2/fold*'), key=lambda item: int(os.path.basename(item).replace('fold','')))
        tprs = []
        aucs = []
        f1s = []
        precisions = []
        aps = []
        accs = []
        roc_plot_data = []
        pr_plot_data = []
        cutoff_plot_data = []
        x_vals = np.linspace(0, 1, 100)
        
        Kfold = len(fold_list)
        
        for fold, path in enumerate(fold_list):
            print(f"Training model based on the data removing fold{fold+1}")
            ## read train set, validation set and test set (ignore the validation set)
            with open(path + '/train_val_test.pkl', 'rb') as infile:
                train_batch, _, test_batch = pickle.load(infile)
                train_data = pd.concat(train_batch).reset_index(drop=True)
                test_data = pd.concat(test_batch).reset_index(drop=True)

            ## prepare the feature vectors
            train_X, train_y = generate_X_and_y(train_data, curie_to_vec_dict, pair_emb='concatenate')
            test_X, test_y = generate_X_and_y(test_data, curie_to_vec_dict, pair_emb='concatenate')

            folder_name = f'RF_trees{args.trees}_maxdepth{args.depth}_{args.m_suffix}'
            try:
                os.mkdir(os.path.join(args.output_folder, folder_name))
            except:
                pass
            try:
                os.mkdir(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation'))
            except:
                pass
            try:
                os.mkdir(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}'))
            except:
                pass
            
            # Sets and fits Random ForestModel
            print('Start training model', flush=True)
            RF_model = ensemble.RandomForestClassifier(class_weight='balanced', max_depth=args.depth, max_leaf_nodes=None, n_estimators=args.trees, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", n_jobs=-1)
            fitModel = RF_model.fit(train_X, train_y)

            # saves the model
            model_name = f'RF_trees{args.trees}_maxdepth{args.depth}_{args.m_suffix}.pt'
            joblib.dump(fitModel, os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', model_name))

            print("")
            print('#### Evaluate best model ####', flush=True)
            test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)
            f1s.append(test_f1score)
            fpr, tpr = test_plot_data['roc_curve']['fpr'], test_plot_data['roc_curve']['tpr']
            roc_plot_data += [(fpr, tpr, test_auc_score, test_f1score)]
            tprs.append(np.interp(x_vals, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(test_auc_score)
            aps.append(test_ap_score)
            accs.append(test_acc)
            precision, recall = test_plot_data['precision_recall_curve']['precision'], test_plot_data['precision_recall_curve']['recall']
            pr_plot_data += [(recall, precision, test_ap_score)]
            precisions.append(np.interp(x_vals, np.flip(recall), np.flip(precision)))
            precisions[-1][0] = 1.0
            cutoff_plot_data += [(test_y_true, test_y_probs)]
            
        ## Saves all evaluation result data for downstream analysis
        all_evaluation_results = dict()        
        all_evaluation_results['f1s'] = f1s
        all_evaluation_results['tprs'] = tprs
        all_evaluation_results['aucs'] = aucs
        all_evaluation_results['accs'] = accs
        all_evaluation_results['precisions'] = precisions
        all_evaluation_results['aps'] = aps
        all_evaluation_results['roc_plot_data'] = roc_plot_data
        all_evaluation_results['pr_plot_data'] = pr_plot_data
        all_evaluation_results['cutoff_plot_data'] = cutoff_plot_data
        with open(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', 'all_evaluation_results.pkl'),'wb') as file:
            pickle.dump(all_evaluation_results, file)
        
        
        ## Print out evaluation metrics for 10 folds
        print("")
        print('##### evaluation metrics for 10 folds #####', flush=True)
        for fold in range(Kfold):
            print(f"Fold {fold+1}: Accuracy {accs[fold]:.5f}; F1 score {f1s[fold]:.5f}; AUC score {aucs[fold]:.5f}; AP score {aps[fold]:.5f}")
        
        
        ## Plots Receiver operating characteristic (ROC) curve
        print("plot Receiver operating characteristic (ROC) curve", flush=True)
        for fold in range(Kfold):
            fpr, tpr, roc_auc, f1 = roc_plot_data[fold]
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.5f, F1 = %0.5f)' % (fold+1, roc_auc, f1))
        
        # Plots the 50/50 line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Coin Flip', alpha=.8)

        # Plots the mean roc curve and mean f1 score
        mean_tpr = np.mean(tprs, axis=0)
        mean_f1 = np.mean(f1s)
        mean_tpr[-1] = 1.0
        mean_auc = met.auc(x_vals, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(x_vals, mean_tpr, color='b', label=u'Mean ROC (AUC = %0.5f \u00B1 %0.5f, \n        \
                        Mean F1 = %0.5f)' % (mean_auc, std_auc, mean_f1),
                    lw=2, alpha=.8)

        # Plots the +- standard deviation for roc curve
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(x_vals, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', 'roc_curve.png')
        plt.savefig(outloc)
        plt.close()
        
        ## Plots Precision Recall (PR) curve
        print("plot Precision Recall (PR) curve", flush=True)
        for fold in range(Kfold):
            recall, precision, ap = pr_plot_data[fold]
            plt.plot(recall, precision, lw=1, alpha=0.3, label='PRC fold %d (AP = %0.5f)' % (fold+1, ap))
        
        # Plots the 50/50 line
        plt.plot([0, 1], [1, 0], linestyle='--', lw=2, color='black', label='Coin Flip', alpha=.8)

        # Plots the mean prc curve and mean ap score
        mean_precision = np.mean(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_auc = np.std(aucs)
        plt.plot(x_vals, mean_precision, color='b', label=u'Mean ROC (AP = %0.5f \u00B1 %0.5f)' % (mean_ap, std_auc), lw=2, alpha=.8)

        # Plots the +- standard deviation for prc curve
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)
        plt.fill_between(x_vals, precisions_lower, precisions_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall (PR) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', 'precision-recall_curve.png')
        plt.savefig(outloc)
        plt.close()
        
        # Plots the cutoff rates for test dataset
        print("## plot the cutoff rates for test dataset", flush=True)
        colors = ["xkcd:dark magenta","xkcd:dark turquoise","xkcd:azure","xkcd:purple blue","xkcd:scarlet",
            "xkcd:orchid", "xkcd:pumpkin", "xkcd:gold", "xkcd:peach", "xkcd:neon green", "xkcd:grey blue"]

        title_post = ["True Positives", "True Negatives"]

        linestyle_tuple = [
             ('loosely dotted',        (0, (1, 10))),
             ('dotted',                (0, (1, 1))),
             ('densely dotted',        (0, (1, 1))),

             ('loosely dashed',        (0, (5, 10))),
             ('dashed',                (0, (5, 5))),
             ('densely dashed',        (0, (5, 1))),

             ('loosely dashdotted',    (0, (3, 10, 1, 10))),
             ('dashdotted',            (0, (3, 5, 1, 5))),
             ('densely dashdotted',    (0, (3, 1, 1, 1))),

             ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        for fold in range(Kfold):
            test_y_true, test_y_probs = cutoff_plot_data[fold]
            probas_tp = test_y_probs[np.where(test_y_true == 1)[0]]
            probas_tn = test_y_probs[np.where(test_y_true == 0)[0]]
            dfs = [pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}), pd.DataFrame({"treat_prob":[pr for pr in probas_tn]})]
            for index, df in enumerate(dfs):
                cutoffs = [x/100 for x in range(101)]
                cutoff_n = [df["treat_prob"][df["treat_prob"] >= cutoff].count()/len(df) for cutoff in cutoffs]
                plt.plot(cutoffs,cutoff_n, color=colors[index], linestyle=linestyle_tuple[fold][-1], label=f"{fold+1} {title_post[index]}")

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Cutoff Prob')
        plt.ylabel('Rate of Postitive Predictions')
        plt.title('Prediction Rates of Treats Class')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', 'test_dataset_rate_cutoff.png')
        plt.savefig(outloc)
        plt.close()

    else:
        print('Running mode only accepts 1 or 2 or 3')
    
    print('#### Program Summary ####')
    end_time = time.time()
    print(f'Total execution time = {end_time - start_time:.3f} sec')
# {"mode":"full","isActive":false}