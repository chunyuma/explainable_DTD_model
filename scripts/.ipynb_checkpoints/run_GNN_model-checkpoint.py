import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from dataset import ProcessedDataset
from model import GAT
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def train(epoch, device):
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Epoch {epoch:03d}')

    total_loss = 0
    for batch_num, train_sampler in enumerate(train_loader):
        for _, n_id, adjs in train_sampler:
            pass
        batch_size = train_batch[batch_num].shape[0]
        adjs = [adj.to(device) for adj in adjs]
        y = torch.tensor(train_batch[batch_num]['y'], dtype=torch.float).to(device)
        # deal with inblance class with weights
        pos = len(torch.where(y == 1)[0])
        neg = len(torch.where(y == 0)[0])
        n_sample = neg + pos
        weights = torch.zeros(n_sample)
        if neg > pos:
            weights[torch.where(y == 1)[0]] = neg/pos
            weights[torch.where(y == 0)[0]] = 1
        elif pos > neg:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = pos/neg
        else:
            weights[torch.where(y == 1)[0]] = 1
            weights[torch.where(y == 0)[0]] = 1
        link = train_batch[batch_num][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
        link = torch.tensor(np.array(link), dtype=torch.long).to(device)
        n_id = n_id.to(device)

        optimizer.zero_grad()
        x_n_id = x[n_id].to(device)
        pred_y= model(x_n_id, adjs, link, n_id)
        train_loss = F.binary_cross_entropy(pred_y, y, weights)
        train_loss.backward()
        optimizer.step()
        total_loss += float(train_loss)
        pbar.update(1)

    train_loss = total_loss / len(train_loader)

    ## evaluate model with validation data
    model.eval()
    with torch.no_grad():
        for batch_num, val_sampler in enumerate(val_loader):
            for _, n_id, val_adj in val_sampler:
                pass
            val_adj = [adj.to(device) for adj in val_adj]
            y = torch.tensor(pairs_val['y'], dtype=torch.float).to(device)
            # deal with inblance class with weights
            pos = len(torch.where(y == 1)[0])
            neg = len(torch.where(y == 0)[0])
            n_sample = neg + pos
            weights = torch.zeros(n_sample)
            if neg > pos:
                weights[torch.where(y == 1)[0]] = neg/pos
                weights[torch.where(y == 0)[0]] = 1
            elif pos > neg:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = pos/neg
            else:
                weights[torch.where(y == 1)[0]] = 1
                weights[torch.where(y == 0)[0]] = 1
            link = pairs_val[['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long).to(device)
            n_id = n_id.to(device)

            x_n_id = x[n_id].to(device)
            pred_y = model(x_n_id, val_adj, link, n_id).detach()
            val_loss = F.binary_cross_entropy(pred_y, y, weights)
            val_loss = float(val_loss)

    print(f"Epoch: {epoch:03d}, Train: {train_loss:.4f}, Val: {val_loss:.4f}", flush=True)

    return [train_loss, val_loss]

def evaluate(loader, device, data_type='train'): # data_type can be 'train', 'val', or 'test'
    model.eval()

    predictions = []
    labels = []
    pbar = tqdm(total=len(loader))

    with torch.no_grad():
        for batch_num, sampler in enumerate(loader):
            for _, n_id, adjs in sampler:
                pass
            adjs = [adj.to(device) for adj in adjs]
            if data_type == 'train':
                link = train_batch[batch_num][['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            elif data_type == 'val':
                link = pairs_val[['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            elif data_type == 'test':
                link = pairs_test[['source','target']].apply(lambda row: [idx_map.get(row[0]),idx_map.get(row[1])], axis=1, result_type='expand').rename(columns={0: "source", 1: "target"})
            link = torch.tensor(np.array(link), dtype=torch.long).to(device)
            n_id = n_id.to(device)

            x_n_id = x[n_id].to(device)
            pred = model(x_n_id, adjs, link, n_id).detach()

            if data_type == 'train':
                label = train_batch[batch_num]['y']
            elif data_type == 'val':
                label = pairs_val['y']
            elif data_type == 'test':
                label = pairs_test['y']

            predictions.append(pred)
            labels.append(label)
            pbar.update(1)

    predictions = torch.hstack(predictions).numpy()
    labels = torch.hstack(labels).numpy()
    
    return roc_auc_score(labels, predictions)


####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from neo4j database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d_path", "--data_path", type=str, help="Data Forlder", default='~/work/explainable_DTD_model/mydata')
    parser.add_argument("-gpu", "--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument("-lr", "--learning_ratio", type=float, help="Learning Ratio", default=0.001)
    parser.add_argument("-epochs", "--num_epochs", type=int, help="Number of epochs to train model", default=50)
    parser.add_argument("-e_size", "--embedding_size", type=int, help="Embedding vertor dimension", default=512)
    parser.add_argument("-batch", "--batch_size", type=int, help="Batch size of training data", default=512)
    parser.add_argument("-layers", "--num_layers", type=int, help="Number of GNN layers to train model", default=3)
    parser.add_argument("-head", "--num_head", type=int, help="Number of head in GAT model", default=8)
    parser.add_argument("-t_v_t_size", "--train_val_test_size", type=str, help="Proportion of training data, validation data and test data", default="[0.8, 0.1, 0.1]")
    parser.add_argument("-dp", "--dropout_p", type=float, help="Drop out proportion", default=0.2)
    parser.add_argument("-o", "--output_folder", type=str, help="The path of output folder", default="~/work/explainable_DTD_model/results")
    args = parser.parse_args()

    raw_edges = pd.read_csv(args.data_path + '/graph_edges.txt', sep='\t', header=0)
    node_info = pd.read_csv(args.data_path + '/graph_nodes_label.txt', sep='\t', header=0)
    tp_pairs = pd.read_csv(args.data_path + '/tp_pairs.txt', sep='\t', header=0)
    tn_pairs = pd.read_csv(args.data_path + '/tn_pairs.txt', sep='\t', header=0)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    num_layers = args.num_layers
    train_val_test_size = eval(args.train_val_test_size)
    dropout_p = args.dropout_p
    lr = args.learning_ratio
    num_head = args.num_head

    processdata_path = args.data_path + '/ProcessedDataset'
    print('Start processing data', flush=True)
    dataset = ProcessedDataset(root=processdata_path, raw_edges=raw_edges, node_info=node_info, tp_pairs=tp_pairs, tn_pairs=tn_pairs, train_val_test_size=train_val_test_size, batch_size=batch_size)
    data = dataset.get_dataset()
    idx_map, category_map = dataset.get_mapfiles()
    train_batch, pairs_val, pairs_test = dataset.get_train_val_test()
    train_loader, val_loader, test_loader = dataset.get_loaders()

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        # torch.backends.cudnn.benchmark = True
    elif args.use_gpu:
        print('No GPU is available in this computer. Use CPU instead.')
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model = GAT(data.feat.shape[1], embedding_size, 1, num_layers=num_layers, dropout_p=dropout_p, num_head = num_head)
    model = model.to(device)

    x = data.feat.to(device)
    total_train_data = (len(train_batch)-1)*batch_size + train_batch[len(train_batch)-1].shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_train_loss = []
    all_val_loss = []

    current_min_train_loss = 100000000
    print('Start training model', flush=True)
    if args.use_gpu and torch.cuda.is_available():
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for epoch in range(num_epochs+20):
            train_loss, val_loss = train(epoch, device)
            all_train_loss += [train_loss]
            all_val_loss += [val_loss]
            if train_loss < current_min_train_loss:
                current_min_train_loss = train_loss
            if epoch >= num_epochs-1 and train_loss <= current_min_train_loss:
                break
    else:
        # with torch.autograd.profiler.profile() as prof:
        for epoch in range(num_epochs+20):
            train_loss, val_loss = train(epoch, device)
            all_train_loss += [train_loss]
            all_val_loss += [val_loss]
            if train_loss < current_min_train_loss:
                current_min_train_loss = train_loss
            if epoch >= num_epochs-1 and train_loss <= current_min_train_loss:
                break

    ## save model and weights
    torch.save({'model_state_dict': model.state_dict()}, args.output_folder + '/train_model.pt')

    ## save data for plotting graph
    epoches = list(range(1,num_epochs+1))
    plotdata = pd.DataFrame(list(zip(epoches,all_train_loss,['train_loss']*num_epochs)) + list(zip(epoches,all_val_loss,['val_loss']*num_epochs)), columns=['epoch', 'loss', 'type'])
    pdata = [epoches, plotdata]

    with open(args.output_folder +'/plotdata.pkl', 'wb') as file:
        pickle.dump(pdata, file)

    print('#### Evaluate model with AUC score ####')
    train_auc = evaluate(train_loader, device, data_type='train')
    val_auc = evaluate(val_loader, device, data_type='val')
    test_auc = evaluate(test_loader, device, data_type='test')
    print(f'Final AUC: Train Auc: {train_auc:.5f}, Val Auc: {val_auc:.5f}, Test Auc: {test_auc:.5f}')