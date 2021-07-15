import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as met
import time
import argparse
import joblib
from glob import glob
from utils import calculate_acc, plot_cutoff, calculate_f1score, calculate_mrr, calculate_hitk


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


def evaluate(model, X, y_true, calculate_metric=True): 

    probas = model.predict_proba(X)[:,1]
    
    if calculate_metric is True:
        
        ## calculate accuracy
        acc = calculate_acc(probas, y_true)
        ## calculate F1 score
        f1score = calculate_f1score(probas,y_true)
        ## calculate AUC
        auc_score = met.roc_auc_score(y_true, probas)
        ## calculate AP (average precision) score
        ap_score = met.average_precision_score(y_true, probas)
    
        plot_data = dict()
        ## generate Receiver operating characteristic (ROC) curve plot data
        fpr, tpr, _ = met.roc_curve(y_true, probas)
        plot_data['roc_curve'] = {'fpr':fpr, 'tpr':tpr}
        
        ## generate Precision Recall Curve plot data
        precision, recall, _ = met.precision_recall_curve(y_true, probas)
        plot_data['precision_recall_curve'] = {'precision':precision, 'recall':recall}
        
        # ## generate detection error tradeoff (DET) curve plot data
        # fpr, fnr, _ = met.det_curve(y_true, probas)
        # plot_data['det_curve'] = {'fpr':fpr, 'fnr':fnr}
        
        return [acc, f1score, auc_score, ap_score, plot_data, y_true, probas]
    
    else:
        
        return [None, None, None, None, None, y_true, probas]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_mode", type=int, help="Model for running model. 1 for normal mode, 2 for crossvalidation", default=1)
    parser.add_argument("--data_path", type=str, help="Data Forlder", default='~/work/GraphSage_logistic_model/data')
    parser.add_argument("--m_suffix", type=str, help="Model Name Suffix", default='default')
    parser.add_argument("--pair_emb", type=str, help="The method for the pair embedding (concatenate or hadamard).", default="concatenate")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/work/GraphSage_logistic_model/results")
    args = parser.parse_args()
    start_time = time.time()

    ## read GraphSage embedding vectors and mapping file
    # node_vec = pd.read_csv(args.data_path + '/graphsage_node_vec.emb', sep = ' ', skiprows=1, header = None, index_col=None)
    # map_file = pd.read_csv(args.data_path + '/graphsage_input/id_map.txt', sep="\t", index_col=None)
    # curie_to_vec = map_file.merge(node_vec, left_on='id', right_on=0)
    # selected_cols = [list(curie_to_vec.columns)[0]] + list(curie_to_vec.columns)[3:]
    # curie_to_vec = curie_to_vec[selected_cols]
    # curie_to_vec_dict = dict()
    # curie_to_vec_dict = {curie_to_vec.loc[index,list(curie_to_vec.columns)[0]]:np.array(curie_to_vec.loc[index,list(curie_to_vec.columns)[1:]]) for index in range(len(curie_to_vec))}
    # del node_vec, map_file, curie_to_vec
    with open(args.data_path + "/all_node_vec.pkl", 'rb') as infile:
        curie_to_vec_dict = pickle.load(infile)
    
    if args.run_mode == 1:
        ## read train set, validation set and test set (ignore the validation set)
        with open(args.data_path + '/mode1/train_val_test_random.pkl', 'rb') as infile:
            train_batch, val_batch, test_batch, random_batch = pickle.load(infile)
            train_data = pd.concat(train_batch).reset_index(drop=True)
            val_data = pd.concat(val_batch).reset_index(drop=True)
            test_data = pd.concat(test_batch).reset_index(drop=True)
            random_data = pd.concat(random_batch).reset_index(drop=True)

        ## prepare the feature vectors
        train_X, train_y = generate_X_and_y(train_data, curie_to_vec_dict, pair_emb='concatenate')
        val_X, val_y = generate_X_and_y(val_data, curie_to_vec_dict, pair_emb='concatenate')
        test_X, test_y = generate_X_and_y(test_data, curie_to_vec_dict, pair_emb='concatenate')
        random_X, random_y = generate_X_and_y(random_data, curie_to_vec_dict, pair_emb='concatenate')

        folder_name = f'logistic_{args.m_suffix}'
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name))
        except:
            pass
        
        # Sets and fits Random ForestModel
        print('Start training model', flush=True)
        logistic_model = LogisticRegression(class_weight='balanced', penalty='none', solver='saga')
        fitModel = logistic_model.fit(train_X, train_y)

        # saves the model
        model_name = f'logistic_{args.m_suffix}.pt'
        joblib.dump(fitModel, os.path.join(args.output_folder, folder_name, model_name))

        print("")
        print('#### Evaluate best model ####', flush=True)
        train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(fitModel, train_X, train_y)
        val_acc, val_f1score, val_auc_score, val_ap_score, val_plot_data, val_y_true, val_y_probs = evaluate(fitModel, val_X, val_y)
        test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)
        test_data['prob'] = test_y_probs
        _, _, _, _, _, random_y_true, random_y_probs = evaluate(fitModel, random_X, random_y, False)
        random_data['prob'] = random_y_probs
        
        test_mrr_score = calculate_mrr(test_data,random_data)
        test_hit1_score = calculate_hitk(test_data,random_data, k=1)
        test_hit10_score = calculate_hitk(test_data,random_data, k=10)
        test_hit20_score = calculate_hitk(test_data,random_data, k=20)
        test_hit50_score = calculate_hitk(test_data,random_data, k=50)     
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
        all_evaluation_results['evaluation_y_true'] = [train_y_true, val_y_true, (test_y_true, random_y_true)]
        all_evaluation_results['evaluation_y_probas'] = [train_y_probs, val_y_probs, (test_y_probs, random_y_probs)]
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
        plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
        pd.DataFrame({"treat_prob":[pr for pr in probas_tn]}),
        pd.DataFrame({'treat_prob':[pr for pr in random_y_probs]})],
        'Prediction Rates of Treats Class for Test dataset',
        os.path.join(args.output_folder, folder_name, 'test_dataset_rate_cutoff.png'),
        ["True Positives", "True Negatives", "Random Pairs"])
        
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

            folder_name = f'logistic_{args.m_suffix}'
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
            logistic_model = LogisticRegression(class_weight='balanced', penalty='none')
            fitModel = logistic_model.fit(train_X, train_y)

            # saves the model
            model_name = f'logistic_{args.m_suffix}.pt'
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