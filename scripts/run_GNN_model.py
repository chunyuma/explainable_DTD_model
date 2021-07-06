import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from dataset import ProcessedDataset, MakeKFoldData, MakeKRandomPairs
from model import GAT
import torch.nn.functional as F
import pickle
import argparse
import torch.cuda.amp as amp
from utils import calculate_acc, format_time, plot_cutoff, calculate_f1score, calculate_mrr, calculate_hitk
import time
import gc
import sklearn.metrics as met
import scipy as sci
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def train(epoch, use_gpu, num_epochs, train_loader, train_batch, val_loader, val_batch):
    print("")
    print(f"======== Epoch {epoch + 1} / {num_epochs} ========")
    print('Training...')
    model.train()
    t0 = time.time()
    
    if use_gpu is True:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])
    for batch_idx, (n_id, adjs) in enumerate(train_loader):
        
        batch_t0 = time.time()
        n_id = n_id[0].to(device)
        adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
        y = torch.tensor(train_batch[batch_idx]['y'], dtype=torch.float).to(device)
        # deal with inblance class with weights
        pos = len(torch.where(y == 1)[0])
        neg = len(torch.where(y == 0)[0])
        n_sample = neg + pos
        weights = torch.zeros(n_sample, dtype=torch.float)
        if neg > pos:
            weights[torch.where(y == 1)[0]] = neg/pos
            weights[torch.where(y == 0)[0]] = 1
        elif pos > neg:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = pos/neg
        else:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = 1
        weights = weights.to(device)
        link = train_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
        link = torch.tensor(np.array(link), dtype=torch.int).to(device)
#         x_n_id = x[n_id]

        optimizer.zero_grad()
        
        if use_gpu is True:
            with amp.autocast(enabled=True): # use mixed precision training
                pred_y= model(all_init_mats, adjs, link, n_id, all_sorted_indexes).to(device)
                # train_loss = F.binary_cross_entropy(pred_y, y, weights)
                train_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
                all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
                all_y = torch.cat([all_y, y.cpu().detach()])
        
            # Scales loss.
            scaler.scale(train_loss).backward()
            
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            
            # Updates the scale for next iteration.
            scaler.update()
        else:
            pred_y= model(all_init_mats, adjs, link, n_id, all_sorted_indexes).to(device)
            # train_loss = F.binary_cross_entropy(pred_y, y, weights)
            train_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
            all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
            all_y = torch.cat([all_y, y.cpu().detach()])
            train_loss.backward()
            optimizer.step()            
        
        total_loss += float(train_loss.detach())
        
        if batch_idx % print_every == 0 and not batch_idx == 0:
            elapsed = format_time(time.time() - batch_t0)
            print(f"Batch {batch_idx} of {len(train_loader)}. This batch costs around: {elapsed}.", flush=True)
        # pbar.update(1)

    train_loss = total_loss / len(train_loader)
    train_acc = calculate_acc(all_pred,all_y)

    ## evaluate model with validation data
    model.eval()
    total_loss = 0
    all_pred = torch.tensor([])
    all_y = torch.tensor([])
    # pbar = tqdm(total=len(val_loaders_path))
    # pbar.set_description(f'Epoch Val{epoch:03d}')
    with torch.no_grad():
        for batch_idx, (n_id, adjs) in enumerate(val_loader):
            n_id = n_id[0].to(device)
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            y = torch.tensor(val_batch[batch_idx]['y'], dtype=torch.float).to(device)
            # deal with inblance class with weights
            pos = len(torch.where(y == 1)[0])
            neg = len(torch.where(y == 0)[0])
            n_sample = neg + pos
            weights = torch.zeros(n_sample, dtype=torch.float)
            if neg > pos:
                weights[torch.where(y == 1)[0]] = neg/pos
                weights[torch.where(y == 0)[0]] = 1
            elif pos > neg:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = pos/neg
            else:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = 1
            weights = weights.to(device)
            link = val_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.int).to(device)
#             x_n_id = x[n_id]
            
            if use_gpu is True:
                with amp.autocast(enabled=True): # use mixed precision training
                    pred_y = model(all_init_mats, adjs, link, n_id, all_sorted_indexes).detach().to(device)
                #   val_loss = F.binary_cross_entropy(pred_y, y, weights)
                    val_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
                    all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
                    all_y = torch.cat([all_y, y.cpu().detach()])
            else:
                pred_y = model(all_init_mats, adjs, link, n_id, all_sorted_indexes).detach().to(device)
                # val_loss = F.binary_cross_entropy(pred_y, y, weights)
                val_loss = F.binary_cross_entropy_with_logits(pred_y, y, weights)
                all_pred = torch.cat([all_pred,torch.sigmoid(pred_y).cpu().detach()])
                all_y = torch.cat([all_y, y.cpu().detach()])

            total_loss += float(val_loss.detach())
            # pbar.update(1)
        
        val_loss = total_loss / len(val_loader)
        val_acc = calculate_acc(all_pred,all_y)

    print(f"Epoch Stat: Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}", flush=True)
    training_time = format_time(time.time() - t0)
    print(f"The total running time of this epoch: {training_time}", flush=True)

    return [train_loss, train_acc, val_loss, val_acc]


def evaluate(loader, use_gpu, batch_data, calculate_metric=True): 
    model.eval()

    predictions = []
    labels = []

    if use_gpu is True:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        for batch_idx, (n_id, adjs) in enumerate(loader):
            n_id = n_id[0].to(device)
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            link = batch_data[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long).to(device)
#             x_n_id = x[n_id]

            if use_gpu is True:
                with amp.autocast(enabled=True): # use mixed precision training
                    pred = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
#                 pred = model(x_n_id, adjs, link, n_id).detach().cpu().numpy()
            else:
                pred = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
#                 pred = model(x_n_id, adjs, link, n_id).detach().cpu().numpy()

            label = np.array(batch_data[batch_idx]['y'])

            predictions.append(pred)
            labels.append(label)
            # pbar.update(1)

    probas = np.hstack(predictions)
    labels = np.hstack(labels)
    
    if calculate_metric is True:
        
        ## calculate accuracy
        acc = calculate_acc(probas,labels)
        ## calculate F1 score
        f1score = calculate_f1score(probas,labels)
        ## calculate AUC
        auc_score = met.roc_auc_score(labels, probas)
        ## calculate AP (average precision) score
        ap_score = met.average_precision_score(labels, probas)
    
        plot_data = dict()
        ## generate Receiver operating characteristic (ROC) curve plot data
        fpr, tpr, _ = met.roc_curve(labels, probas)
        plot_data['roc_curve'] = {'fpr':fpr, 'tpr':tpr}
        
        ## generate Precision Recall Curve plot data
        precision, recall, _ = met.precision_recall_curve(labels, probas)
        plot_data['precision_recall_curve'] = {'precision':precision, 'recall':recall}
        
        ## generate detection error tradeoff (DET) curve plot data
        fpr, fnr, _ = met.det_curve(labels, probas)
        plot_data['det_curve'] = {'fpr':fpr, 'fnr':fnr}
        
        return [acc, f1score, auc_score, ap_score, plot_data, labels, probas]
    
    else:
        
        return [None, None, None, None, None, labels, probas]
    
# def predict_res(loader, batch, use_gpu=True):
#     model.eval()

#     preds = []
#     probas = []
#     labels = []

#     with torch.no_grad():
#         for batch_idx, (n_id, adjs) in enumerate(loader):
#             n_id = n_id[0]
#             adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
#             link = batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
#             link = torch.tensor(np.array(link), dtype=torch.long)

#             if use_gpu is True:
#                 with amp.autocast(enabled=True): # use mixed precision training
#                     proba = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
#             else:
#                 proba = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()


#             label = np.array(batch[batch_idx]['y'])

#             probas.append(proba)
#             preds.append(np.array([1 if value>=0.5 else 0 for value in proba]))
#             labels.append(label)
#             # pbar.update(1)

#     probas = np.hstack(probas)
#     preds = np.hstack(preds)
#     labels = np.hstack(labels)
    
#     return [labels, preds, probas]    


####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_mode", type=int, help="Model for running model. 1 for normal mode, 2 for crossvalidation", default=1)
    parser.add_argument("--data_path", type=str, help="Data Forlder", default='~/work/explainable_DTD_model/data')
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument("--use_multiple_gpu", action="store_true", help="Use all GPUs on computer to train model", default=False)
    parser.add_argument("--seed", type=float, help="Manually set initial seed for pytorch", default=1020)
    parser.add_argument("--learning_ratio", type=float, help="Learning ratio", default=0.001)
    parser.add_argument("--init_emb_size", type=int, help="Initial embedding", default=100)
    parser.add_argument("--use_known_embedding", action="store_true", help="Use known inital embeeding", default=False)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train model", default=50)
    parser.add_argument("--Kfold", type=int, help="Number of fold", default=10)
    parser.add_argument("--mrr_hk_n", type=int, help="Number of random pair for MRR and H@K", default=5000)
    parser.add_argument("--n_random_pairs", type=int, help="Number of random pairs for mode 3", default=20000)
    parser.add_argument("--emb_size", type=int, help="Embedding vertor dimension", default=512)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=512)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers to train model", default=3)
    parser.add_argument("--patience", type=int, help="Number of epochs with no improvement after which learning rate will be reduced", default=10)
    parser.add_argument("--factor", type=float, help="The factor for learning rate to be reduced", default=0.1)
    parser.add_argument("--early_stop_n", type=int, help="Early stop if validation loss doesn't further decrease after n step", default=50)
    parser.add_argument("--num_head", type=int, help="Number of head in GAT model", default=8)
    parser.add_argument("--print_every", type=int, help="How often to print training batch elapsed time", default=10)
    parser.add_argument("--train_val_test_size", type=str, help="Proportion of training data, validation data and test data", default="[0.8, 0.1, 0.1]")
    parser.add_argument("--dropout_p", type=float, help="Drop out proportion", default=0.2)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default="~/work/explainable_DTD_model/results")
    args = parser.parse_args()
    start_time = time.time()
    
    raw_edges = pd.read_csv(args.data_path + '/graph_edges.txt', sep='\t', header=0)
    node_info = pd.read_csv(args.data_path + '/graph_nodes_label.txt', sep='\t', header=0)
    tp_pairs = pd.read_csv(args.data_path + '/tp_pairs.txt', sep='\t', header=0)
    tn_pairs = pd.read_csv(args.data_path + '/tn_pairs.txt', sep='\t', header=0)
    all_known_tp_pairs = pd.read_csv(args.data_path + '/all_known_tps.txt', sep='\t', header=0)
    if args.use_known_embedding:
        with open(args.data_path + '/known_int_emb_dict.pkl','rb') as infile:
            known_int_emb_dict = pickle.load(infile)
    else:
        known_int_emb_dict = None
    
    torch.manual_seed(args.seed)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_size = args.emb_size
    num_layers = args.num_layers
    train_val_test_size = eval(args.train_val_test_size)
    dropout_p = args.dropout_p
    lr = args.learning_ratio
    num_head = args.num_head
    init_emb_size = args.init_emb_size
    print_every = args.print_every
    patience = args.patience
    factor = args.factor
    early_stop_n = args.early_stop_n
    Kfold = args.Kfold
    n_random_pairs = args.n_random_pairs
    mrr_hk_n = args.mrr_hk_n

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        use_multiple_gpu = args.use_multiple_gpu
        torch.cuda.reset_peak_memory_stats()
    elif args.use_gpu:
        print('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
    else:
        use_gpu = False
    
    if args.run_mode == 1:

        processdata_path = os.path.join(args.data_path, f'ProcessedDataset_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = ProcessedDataset(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, all_known_tp_pairs=all_known_tp_pairs, train_val_test_size=train_val_test_size, batch_size=batch_size, layers=num_layers, dim=init_emb_size, known_int_emb_dict=known_int_emb_dict, N=mrr_hk_n)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, tp_pairs, tn_pairs, all_known_tp_pairs ## remove the unused varaibles to release memory
        if args.use_known_embedding:
            del known_int_emb_dict
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        train_batch, val_batch, test_batch, random_batch = dataset.get_train_val_test_random()
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        test_loader = dataset.get_test_loader()
        random_loader = dataset.get_random_loader()
        init_emb = data.feat
        type_init_emb_size = [init_emb[key][0].shape[1] for key,value in init_emb.items()]
        
        model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
        folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:05d}_patience{patience}_factor{factor}'
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name))
        except:
            pass
        writer = SummaryWriter(log_dir=os.path.join(args.output_folder, folder_name, 'tensorboard_runs'))
        
        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=0.0001, threshold_mode='rel')
        if use_gpu:
            scaler = amp.GradScaler(enabled=True) # scaler for mixed precision training
        all_train_loss = []
        all_val_loss = []
        all_train_acc = []
        all_val_acc = []
        
        current_min_val_loss = float('inf')
        model_state_dict = None
        count = 0
        print('Start training model', flush=True)
        for epoch in trange(num_epochs):
            if count > early_stop_n:
                break
            train_loss, train_acc, val_loss, val_acc = train(epoch, use_gpu, num_epochs, train_loader, train_batch, val_loader, val_batch)
            scheduler.step(val_loss)
            all_train_loss += [train_loss]
            all_val_loss += [val_loss]
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            all_train_acc += [train_acc]
            all_val_acc += [val_acc]
            writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            count += 1
            if val_loss < current_min_val_loss:
                count = 0
                current_min_val_loss = val_loss
                model_state_dict = model.state_dict()
                model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch+1:05d}_val_loss{current_min_val_loss:.5f}_patience{patience}_factor{factor}.pt'   

        writer.close()
        ## Saves model and weights
        torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, model_name))
        
        ## Saves data for plotting graph
        epoches = list(range(1,num_epochs+1))
        plotdata_loss = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
        plotdata_acc = pd.DataFrame(list(zip(epoches,all_train_acc,['train_acc']*num_epochs)) + list(zip(epoches,all_val_acc,['val_acc']*num_epochs)), columns=['epoch', 'acc', 'type'])
        acc_loss_plotdata = [epoches, plotdata_loss, plotdata_acc]

        with open(os.path.join(args.output_folder, folder_name, 'acc_loss_plotdata.pkl'), 'wb') as file:
            pickle.dump(acc_loss_plotdata, file)
    
        print("")
        print('#### Load in the best model', flush=True)
        model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
        model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, model_name))['model_state_dict'])

        print("")
        print('#### Evaluate best model ####', flush=True)
        train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(train_loader, use_gpu, train_batch)
        val_acc, val_f1score, val_auc_score, val_ap_score, val_plot_data, val_y_true, val_y_probs = evaluate(val_loader, use_gpu, val_batch)
        test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(test_loader, use_gpu, test_batch)
        test_data = pd.concat(test_batch)
        test_data['prob'] = test_y_probs
        _, _, _, _, _, random_y_true, random_y_probs = evaluate(random_loader, use_gpu, random_batch, False)
        random_data = pd.concat(random_batch)
        random_data['prob'] = random_y_probs
        test_mrr_score = calculate_mrr(test_data,random_data)
        test_hit1_score = calculate_hitk(test_data,random_data, k=1)
        test_hit10_score = calculate_hitk(test_data,random_data, k=10)
        test_hit20_score = calculate_hitk(test_data,random_data, k=20)
        test_hit50_score = calculate_hitk(test_data,random_data, k=50)     
        print(f'Final AUC: Train Auc: {train_auc_score:.5f}, Validation Auc: {val_auc_score:.5f}, Test Auc: {test_auc_score:.5f}')
        print(f'Final Accuracy: Train Accuracy: {train_acc:.5f}, Validation Accuracy: {val_acc:.5f}, Test Accuracy: {test_acc:.5f}')
        print(f'Final F1 score: Train F1score: {train_f1score:.5f}, Validation F1score: {val_f1score:.5f}, Test F1score: {test_f1score:.5f}')
        print(f'Final AP score: Train APscore: {train_ap_score:.5f}, Validation APscore: {val_ap_score:.5f}, Test APscore: {test_ap_score:.5f}')
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

        ## Plots Detection Error Tradeoff (DET) curve
        print("plot Detection Error Tradeoff (DET) curve", flush=True)
        for datatype in ['train', 'val', 'test']:
            temp_plot_data = eval(datatype+'_plot_data')
            fpr, fnr = temp_plot_data['det_curve']['fpr'], temp_plot_data['det_curve']['fnr']
            plt.plot(fpr, fnr, lw=2, label=f'{datatype.capitalize()} Data')

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, 'det_curve.png')
        plt.savefig(outloc)
        plt.close()

        ## Plots Rate Cutoff for test dataset
        print("plot Rate Cutoff for test dataset", flush=True)        
        probas_tp = test_y_probs[np.where(test_y_true == 1)[0]]
        probas_tn = test_y_probs[np.where(test_y_true == 0)[0]]
        plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
        pd.DataFrame({"treat_prob":[pr for pr in probas_tn]}),
        pd.DataFrame({'treat_prob':[pr for pr in random_y_probs]})],
        'Prediction Rates of Treats Class for Test dataset',
        os.path.join(args.output_folder, folder_name, 'test_dataset_rate_cutoff.png'),
        ["True Positives", "True Negatives", "Random Pairs"])
        
        
    elif args.run_mode == 2:

        processdata_path = os.path.join(args.data_path, f'crossvalidation_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = MakeKFoldData(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, all_known_tp_pairs=all_known_tp_pairs, K=Kfold, batch_size=batch_size, layers=num_layers, dim=init_emb_size, known_int_emb_dict=known_int_emb_dict, N=mrr_hk_n)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, tp_pairs, tn_pairs, all_known_tp_pairs ## remove the unused varaibles to release memory
        if args.use_known_embedding:
            del known_int_emb_dict
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        init_emb = data.feat
        type_init_emb_size = [init_emb[key][0].shape[1] for key,value in init_emb.items()]
        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
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
                                                                        
        for fold in range(Kfold):
            print(f"Training model based on the data removing fold{fold+1}")
            train_batch, val_batch, test_batch = dataset.get_train_val_test(fold+1)
#             train_batch, val_batch, test_batch, random_batch = dataset.get_train_val_test_random(fold+1)
            train_loader = dataset.get_train_loader(fold+1)
            val_loader = dataset.get_val_loader(fold+1)
            test_loader = dataset.get_test_loader(fold+1)
#             random_loader = dataset.get_random_loader(fold+1)
            
            model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
            folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:05d}_patience{patience}_factor{factor}'
            
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
            try:
                os.mkdir(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', 'tensorboard_runs'))   
            except:
                pass
            writer = SummaryWriter(log_dir=os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold}', 'tensorboard_runs'))
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=0.0001, threshold_mode='rel')
            if use_gpu:
                scaler = amp.GradScaler(enabled=True)
            all_train_loss = []
            all_val_loss = []
            all_train_acc = []
            all_val_acc = []

            current_min_val_loss = float('inf')
            model_state_dict = None
            count = 0
            print('Start training model', flush=True)
            for epoch in trange(num_epochs):
                if count > early_stop_n:
                    break
                train_loss, train_acc, val_loss, val_acc = train(epoch, use_gpu, num_epochs, train_loader, train_batch, val_loader, val_batch)
#                 scheduler.step(val_loss)
                all_train_loss += [train_loss]
                all_val_loss += [val_loss]
                writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                all_train_acc += [train_acc]
                all_val_acc += [val_acc]
                writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                count += 1
                if val_loss < current_min_val_loss:
                    count = 0
                    current_min_val_loss = val_loss
                    model_state_dict = model.state_dict()
                    model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch:05d}_val_loss{current_min_val_loss:.5f}_patience{patience}_factor{factor}.pt'   

            writer.close()
            ## Saves model and weights
            torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', model_name))
        
            
            print("")
            print('#### Load in the best model', flush=True)
            model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
            model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', model_name))['model_state_dict'])

            print("")
            print('#### Evaluate best model ####', flush=True)
            test_acc, test_f1score, test_auc_score, test_ap_score, test_plot_data, test_y_true, test_y_probs = evaluate(test_loader, use_gpu, test_batch)
#             test_data = pd.concat(test_batch)
#             test_data['prob'] = test_y_probs
#             _, _, _, _, _, random_y_true, random_y_probs = evaluate(random_loader, use_gpu, random_batch, False)
#             random_data = pd.concat(random_batch)
#             random_data['prob'] = random_y_probs
#             test_mrr_score = calculate_mrr(test_data,random_data)
#             test_hit1_score = calculate_hitk(test_data,random_data, k=1)
#             test_hit10_score = calculate_hitk(test_data,random_data, k=10)
#             test_hit20_score = calculate_hitk(test_data,random_data, k=20)
#             test_hit50_score = calculate_hitk(test_data,random_data, k=50) 
      
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
        
        
    elif args.run_mode == 3:

        processdata_path = os.path.join(args.data_path, f'randompairs_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = MakeKRandomPairs(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, batch_size=batch_size, layers=num_layers, dim=init_emb_size, N=n_random_pairs)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, tp_pairs, tn_pairs ## remove the unused varaibles to release memory
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        train_batch, val_batch, tp_batch, tn_batch, rp_batch = dataset.get_batch_set()
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        tp_loader = dataset.get_tp_loader()
        tn_loader = dataset.get_tn_loader()
        rp_loader = dataset.get_rp_loader()
        init_emb = data.feat
        type_init_emb_size = [init_emb[key][0].shape[1] for key,value in init_emb.items()]
        
        model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
        folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:05d}_patience{patience}_factor{factor}'
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name))
        except:
            pass
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name, 'randompairs'))
        except:
            pass
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name, 'randompairs', 'tensorboard_runs'))       
        except:
            pass
        writer = SummaryWriter(log_dir=os.path.join(args.output_folder, folder_name, 'randompairs', 'tensorboard_runs'))

        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=0.0001, threshold_mode='rel')
        if use_gpu:
            scaler = amp.GradScaler(enabled=True) # scaler for mixed precision training
        all_train_loss = []
        all_val_loss = []
        all_train_acc = []
        all_val_acc = []
        
        current_min_val_loss = float('inf')
        model_state_dict = None
        count = 0
        print('Start training model', flush=True)
        for epoch in trange(num_epochs):
            if count > early_stop_n:
                break
            train_loss, train_acc, val_loss, val_acc = train(epoch, use_gpu, num_epochs, train_loader, train_batch, val_loader, val_batch)
#             scheduler.step(val_loss)
            all_train_loss += [train_loss]
            all_val_loss += [val_loss]
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            all_train_acc += [train_acc]
            all_val_acc += [val_acc]
            writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            count += 1
            if val_loss < current_min_val_loss:
                count = 0
                current_min_val_loss = val_loss
                model_state_dict = model.state_dict()
                model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch:05d}_val_loss{current_min_val_loss:.5f}_patience{patience}_factor{factor}.pt'
                
        writer.close()
        ## Saves model and weights
        torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, 'randompairs', model_name))
        
        ## Saves data for plotting graph
        epoches = list(range(1,num_epochs+1))
        plotdata_loss = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
        plotdata_acc = pd.DataFrame(list(zip(epoches,all_train_acc,['train_acc']*num_epochs)) + list(zip(epoches,all_val_acc,['val_acc']*num_epochs)), columns=['epoch', 'acc', 'type'])
        acc_loss_plotdata = [epoches, plotdata_loss, plotdata_acc]

        with open(os.path.join(args.output_folder, folder_name, 'randompairs', 'acc_loss_plotdata.pkl'), 'wb') as file:
            pickle.dump(acc_loss_plotdata, file)
    
        print("")
        print('#### Load in the best model', flush=True)
        model = GAT(type_init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head, use_gpu=use_gpu, use_multiple_gpu=use_multiple_gpu)
        model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, 'randompairs', model_name))['model_state_dict'])
        
        # Gets random pairs cutoff rates
        _, _, _, _, _, _, probas_rand = evaluate(rp_loader, use_gpu, rp_batch, False)
        
        # Gets true positive cutoff rates
        _, _, _, _, _, _, probas_tp = evaluate(tp_loader, use_gpu, tp_batch, False)

        # Gets true negative cutoff rates
        _, _, _, _, _, _, probas_tn = evaluate(tn_loader, use_gpu, tn_batch, False)
        
        # Plots the cutoff rates together
        plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_rand]}),
                pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
                pd.DataFrame({"treat_prob":[pr for pr in probas_tn]})],
                'Prediction Rates of Treats Class for all datasets',
                os.path.join(args.output_folder, folder_name, 'randompairs', 'all_dataset_rate_cutoff.png'),
                ["Random Pairs",
                "True Positives", 
                "True Negatives"])


        print("")
        print('#### Evaluate best model with AUC score ####')
        train_acc, train_f1score, train_auc_score, train_ap_score, train_plot_data, train_y_true, train_y_probs = evaluate(train_loader, use_gpu, train_batch)
        val_acc, val_f1score, val_auc_score, val_ap_score, val_plot_data, val_y_true, val_y_probs = evaluate(val_loader, use_gpu, val_batch)
        print(f'Final AUC: Train Auc: {train_auc_score:.5f}, Validation Auc: {val_auc_score:.5f}')
        print(f'Final Accuracy: Train Accuracy: {train_acc:.5f}, Validation Accuracy: {val_acc:.5f}')
        print(f'Final F1 score: Train F1score: {train_f1score:.5f}, Validation F1score: {val_f1score:.5f}')
        print(f'Final AP score: Train APscore: {train_ap_score:.5f}, Validation APscore: {val_ap_score:.5f}')
        
        ## Saves all evaluation result data for downstream analysis
        all_evaluation_results = dict()
        all_evaluation_results['evaluation_acc_score'] = [train_acc, val_acc]
        all_evaluation_results['evaluation_f1_score'] = [train_f1score, val_f1score]
        all_evaluation_results['evaluation_auc_score'] = [train_auc_score, val_auc_score]
        all_evaluation_results['evaluation_ap_score'] = [train_ap_score, val_ap_score]
        all_evaluation_results['evaluation_plot_data'] = [train_plot_data, val_plot_data]
        all_evaluation_results['evaluation_y_true'] = [train_y_true, val_y_true]
        all_evaluation_results['evaluation_y_probas'] = [train_y_probs, val_y_probs]
        with open(os.path.join(args.output_folder, folder_name, 'randompairs', 'all_evaluation_results.pkl'),'wb') as file:
            pickle.dump(all_evaluation_results, file)
        
        ## Plots Receiver operating characteristic (ROC) curve
        print("plot Receiver operating characteristic (ROC) curve", flush=True)
        for datatype in ['train', 'val']:
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
        for datatype in ['train', 'val']:
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

        ## Plots Detection Error Tradeoff (DET) curve
        print("plot Detection Error Tradeoff (DET) curve", flush=True)
        for datatype in ['train', 'val']:
            temp_plot_data = eval(datatype+'_plot_data')
            fpr, fnr = temp_plot_data['det_curve']['fpr'], temp_plot_data['det_curve']['fnr']
            plt.plot(fpr, fnr, lw=2, label=f'{datatype.capitalize()} Data')

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        outloc = os.path.join(args.output_folder, folder_name, 'det_curve.png')
        plt.savefig(outloc)
        plt.close()

        ## Plots Rate Cutoff for validation dataset
        print("plot Rate Cutoff for validation dataset", flush=True)
        probas_tp = val_y_probs[np.where(val_y_true == 1)[0]]
        probas_tn = val_y_probs[np.where(val_y_true == 0)[0]] 
        plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
        pd.DataFrame({"treat_prob":[pr for pr in probas_tn]})],
        'Prediction Rates of Treats Class for Validation dataset',
        os.path.join(args.output_folder, folder_name, 'val_dataset_rate_cutoff.png'),
        ["True Positives", "True Negatives"])
        
    else:
        print('Running mode only accepts 1 or 2 or 3')
        
    print('#### Program Summary ####')
    end_time = time.time()
    print(f'Total execution time = {end_time - start_time:.3f} sec')
    if use_gpu is True:
        for index in range(torch.cuda.device_count()):
            print(f'Max memory used by tensors = {torch.cuda.max_memory_allocated(index)} bytes for GPU:{index}')
            print(f'Max memory managed by caching allocator = {torch.cuda.max_memory_reserved(index)} bytes for GPU:{index}')
        gc.collect()
        torch.cuda.empty_cache() 
    











    

