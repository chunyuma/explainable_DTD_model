import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from dataset import ProcessedDataset, MakeKFoldData, MakeKRandomPairs
from model import GAT
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pickle
import argparse
import torch.cuda.amp as amp
from utils import calculate_acc, format_time, plot_cutoff
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
        n_id = n_id[0]
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
        link = torch.tensor(np.array(link), dtype=torch.int)
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
            print(f"Batch {batch_idx} of {len(train_loader)}. Elapsed: {elapsed}.", flush=True)
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
            n_id = n_id[0]
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
            link = torch.tensor(np.array(link), dtype=torch.int)
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

    print(f"Single Epoch Stat: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)
    training_time = format_time(time.time() - t0)
    print(f"The total running time of this epoch: {training_time}", flush=True)

    return [train_loss, train_acc, val_loss, val_acc]


def evaluate(loader, use_gpu, data_type = 'train'): 
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for batch_idx, (n_id, adjs) in enumerate(loader):
            n_id = n_id[0]
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            if data_type == 'train':
                link = train_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            elif data_type == 'val':
                link = val_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            elif data_type == 'test':
                link = test_batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long)
#             x_n_id = x[n_id]

            if use_gpu is True:
                with amp.autocast(enabled=True): # use mixed precision training
                    pred = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
#                 pred = model(x_n_id, adjs, link, n_id).detach().cpu().numpy()
            else:
                pred = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
#                 pred = model(x_n_id, adjs, link, n_id).detach().cpu().numpy()

            if data_type == 'train':
                label = np.array(train_batch[batch_idx]['y'])
            elif data_type == 'val':
                label = np.array(val_batch[batch_idx]['y'])
            elif data_type == 'test':
                label = np.array(test_batch[batch_idx]['y'])

            predictions.append(pred)
            labels.append(label)
            # pbar.update(1)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    
    return roc_auc_score(labels, predictions)


def predict_res(loader, batch, use_gpu=True):
    model.eval()

    preds = []
    probas = []
    labels = []

    with torch.no_grad():
        for batch_idx, (n_id, adjs) in enumerate(loader):
            n_id = n_id[0]
            adjs = [(adj[0][0],(int(adj[1][0]),int(adj[1][1]))) for adj in adjs]
            link = batch[batch_idx][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long)

            if use_gpu is True:
                with amp.autocast(enabled=True): # use mixed precision training
                    proba = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()
            else:
                proba = torch.sigmoid(model(all_init_mats, adjs, link, n_id, all_sorted_indexes)).detach().cpu().numpy()


            label = np.array(batch[batch_idx]['y'])

            probas.append(proba)
            preds.append(np.array([1 if value>=0.5 else 0 for value in proba]))
            labels.append(label)
            # pbar.update(1)

    probas = np.hstack(probas)
    preds = np.hstack(preds)
    labels = np.hstack(labels)
    
    return [labels, preds, probas]    


####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_mode", type=int, help="Model for running model. 1 for normal mode, 2 for crossvalidation", default=1)
    parser.add_argument("--data_path", type=str, help="Data Forlder", default='~/work/explainable_DTD_model/mydata')
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument("--learning_ratio", type=float, help="Learning ratio", default=0.001)
    parser.add_argument("--init_emb_size", type=int, help="Initial embedding", default=100)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train model", default=50)
    parser.add_argument("--Kfold", type=int, help="Number of fold", default=10)
    parser.add_argument("--emb_size", type=int, help="Embedding vertor dimension", default=512)
    parser.add_argument("--batch_size", type=int, help="Batch size of training data", default=512)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers to train model", default=3)
    parser.add_argument("--patience", type=int, help="Number of epochs with no improvement after which learning rate will be reduced", default=10)
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
    treat_pairs = pd.read_csv(args.data_path + '/treat.txt', sep='\t', header=0)
    not_treat_pairs = pd.read_csv(args.data_path + '/not_treat.txt', sep='\t', header=0)
    contraindicated_for_pairs = pd.read_csv(args.data_path + '/contraindicated_for.txt', sep='\t', header=0)

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
    early_stop_n = args.early_stop_n
    Kfold = args.Kfold

    if args.use_gpu and torch.cuda.is_available():
        use_gpu = True
        torch.cuda.reset_peak_memory_stats()
    elif args.use_gpu:
        print('No GPU is detected in this computer. Use CPU instead.')
        use_gpu = False
    else:
        use_gpu = False
    
    if args.run_mode == 1:

        processdata_path = os.path.join(args.data_path, f'ProcessedDataset_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = ProcessedDataset(root=processdata_path, raw_edges=raw_edges, node_info=node_info, treat_pairs=treat_pairs, not_treat_pairs=not_treat_pairs, contraindicated_for_pairs=contraindicated_for_pairs, train_val_test_size=train_val_test_size, batch_size=batch_size, layers=num_layers, dim=init_emb_size)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, treat_pairs, not_treat_pairs, contraindicated_for_pairs ## remove the unused varaibles to release memory
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        train_batch, val_batch, test_batch = dataset.get_train_val_test()
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()
        test_loader = dataset.get_test_loader()
        
        model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
        folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:04d}_patience{patience}'
        try:
            os.mkdir(os.path.join(args.output_folder, folder_name))
        except:
            pass
        writer = SummaryWriter(log_dir=os.path.join(args.output_folder, folder_name, 'tensorboard_runs'))
        
        init_emb = data.feat
        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, threshold=0.0001, threshold_mode='rel')
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
                model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch:04d}_val_loss{current_min_val_loss:.3f}_patience{patience}.pt'   

        writer.close()
        ## save model and weights
        torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, model_name))
        
        ## save data for plotting graph
        epoches = list(range(1,num_epochs+1))
        plotdata_loss = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
        plotdata_acc = pd.DataFrame(list(zip(epoches,all_train_acc,['train_acc']*num_epochs)) + list(zip(epoches,all_val_acc,['val_acc']*num_epochs)), columns=['epoch', 'acc', 'type'])
        pdata = [epoches, plotdata_loss, plotdata_acc]

        with open(os.path.join(args.output_folder, folder_name, 'plotdata.pkl'), 'wb') as file:
            pickle.dump(pdata, file)
    
        print("")
        print('#### Load in the best model', flush=True)
        model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
        model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, model_name))['model_state_dict'])

        print("")
        print('#### Evaluate model with AUC score ####')
        train_auc = evaluate(train_loader, use_gpu, data_type = 'train')
        val_auc = evaluate(val_loader, use_gpu, data_type = 'val')
        test_auc = evaluate(test_loader, use_gpu, data_type = 'test')
        print(f'Final AUC: Train Auc: {train_auc:.5f}, Val Auc: {val_auc:.5f}, Test Auc: {test_auc:.5f}')

    elif args.run_mode == 2:

        processdata_path = os.path.join(args.data_path, f'crossvalidation_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = MakeKFoldData(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, K=Kfold, batch_size=batch_size, layers=num_layers, dim=init_emb_size)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, tp_pairs, tn_pairs ## remove the unused varaibles to release memory
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        init_emb = data.feat
        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
        tprs = []
        aucs = []
        f1s = []
        mean_fpr = np.linspace(0, 1, 100)
                                                                        
        for fold in range(Kfold):
            print(f"Training model based on the data removing fold{fold+1}")
            train_batch, val_batch, test_batch = dataset.get_train_val_test(fold+1)
            train_loader = dataset.get_train_loader(fold+1)
            val_loader = dataset.get_val_loader(fold+1)
            test_loader = dataset.get_test_loader(fold+1)
            
            model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
            folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:04d}_patience{patience}'
            
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, threshold=0.0001, threshold_mode='rel')
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
                    model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch:04d}_val_loss{current_min_val_loss:.3f}_patience{patience}.pt'   

            writer.close()
            ## save model and weights
            torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', model_name))
        
            
            print("")
            print('#### Load in the best model', flush=True)
            model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
            model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'fold{fold+1}', model_name))['model_state_dict'])

            labels, preds, probas = predict_res(test_loader, test_batch, use_gpu=use_gpu)
            f1 = met.f1_score(labels, preds, average='binary')
            f1s.append(f1)
            fpr, tpr, thresholds = met.roc_curve(labels, probas)
            tprs.append(sci.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = met.auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f, F1 = %0.4f)' % (fold+1, roc_auc, f1))

        
        # Plots the 50/50 line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Coin Flip', alpha=.8)

        # Finds and plots the mean roc curve and mean f1 score
        mean_tpr = np.mean(tprs, axis=0)
        mean_f1 = np.mean(f1s)
        mean_tpr[-1] = 1.0
        mean_auc = met.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=u'Mean ROC (AUC = %0.4f \u00B1 %0.4f, \n        \
                        Mean F1 = %0.4f)' % (mean_auc, std_auc, mean_f1),
                    lw=2, alpha=.8)

        # Finds and plots the +- standard deviation for roc curve
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        # Sets legend, limits, labels, and displays plot
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        outloc = os.path.join(args.output_folder, folder_name, f'{Kfold}foldcrossvalidation', f'{Kfold}crossvalidation.png')
        plt.savefig(outloc)
        plt.close()
        
    elif args.run_mode == 3:

        processdata_path = os.path.join(args.data_path, f'randompairs_initemb{init_emb_size}_batch{batch_size}_layer{num_layers}')
        print('Start pre-processing data', flush=True)
        dataset = MakeKRandomPairs(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, batch_size=batch_size, layers=num_layers, dim=init_emb_size)
        print('Pre-processing data completed', flush=True)
        del raw_edges, node_info, tp_pairs, tn_pairs ## remove the unused varaibles to release memory
        data = dataset.get_dataset()
        idx_map, id_to_type, typeid = dataset.get_mapfiles()
        train_batch, test_batch, tp_batch, tn_batch, rp_batch = dataset.get_batch_set()
        train_loader = dataset.get_train_loader()
        test_loader = dataset.get_test_loader()
        tp_loader = dataset.get_tp_loader()
        tn_loader = dataset.get_tn_loader()
        rp_loader = dataset.get_rp_loader()
        
        model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
        folder_name = f'batchsize{batch_size}_initemb{init_emb_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{num_epochs:04d}_patience{patience}'
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

        init_emb = data.feat
        all_sorted_indexes = torch.hstack([init_emb[key][1] for key,value in init_emb.items()]).sort().indices
        all_init_mats = [init_emb[key][0] for key,value in init_emb.items()]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, threshold=0.0001, threshold_mode='rel')
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
            train_loss, train_acc, val_loss, val_acc = train(epoch, use_gpu, num_epochs, train_loader, train_batch, test_loader, test_batch)
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
                model_name = f'GAT_batchsize{batch_size}_embeddingsize{embedding_size}_layers{num_layers}_numhead{num_head}_lr{lr}_epoch{epoch:04d}_val_loss{current_min_val_loss:.3f}_patience{patience}.pt'
                
        writer.close()
        ## save model and weights
        torch.save({'model_state_dict': model_state_dict}, os.path.join(args.output_folder, folder_name, 'randompairs', model_name))
        
        ## save data for plotting graph
        epoches = list(range(1,num_epochs+1))
        plotdata_loss = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
        plotdata_acc = pd.DataFrame(list(zip(epoches,all_train_acc,['train_acc']*num_epochs)) + list(zip(epoches,all_val_acc,['val_acc']*num_epochs)), columns=['epoch', 'acc', 'type'])
        pdata = [epoches, plotdata_loss, plotdata_acc]

        with open(os.path.join(args.output_folder, folder_name, 'randompairs', 'plotdata.pkl'), 'wb') as file:
            pickle.dump(pdata, file)
    
        print("")
        print('#### Load in the best model', flush=True)
        model = GAT(init_emb_size, embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_node_types=len(typeid), num_head = num_head, use_gpu=use_gpu)
        model.load_state_dict(torch.load(os.path.join(args.output_folder, folder_name, 'randompairs', model_name))['model_state_dict'])
        
        # Get random pairs cutoff rates
        _, _, probas_rand = predict_res(rp_loader, rp_batch, use_gpu=use_gpu)
        
        # Get true positive cutoff rates
        _, _, probas_tp = predict_res(tp_loader, tp_batch, use_gpu=use_gpu)  

        # Get true negative cutoff rates
        _, _, probas_tn = predict_res(tn_loader, tn_batch, use_gpu=use_gpu)
        
        # Plot the cutoff rates together
        plot_cutoff([pd.DataFrame({"treat_prob":[pr for pr in probas_rand]}),
                pd.DataFrame({"treat_prob":[pr for pr in probas_tp]}),
                pd.DataFrame({"treat_prob":[pr for pr in probas_tn]})],
                os.path.join(args.output_folder, folder_name, 'randompairs'),
                ["Random Pairs",
                "True Positives", 
                "True Negatives"])
        
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
    











    

