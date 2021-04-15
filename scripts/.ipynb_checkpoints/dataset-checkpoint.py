import pandas as pd
import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, NeighborSampler
import pickle
import math
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import multiprocessing
from glob import glob

class ProcessedDataset(InMemoryDataset):
    def __init__(self, root, raw_edges, node_info, tp_pairs, tn_pairs, transform=None, pre_transform=None, train_val_test_size=[0.8, 0.1, 0.1], batch_size=512):
        try:
            assert sum(train_val_test_size)==1
        except AssertionError:
            print("The sum of percents in train_val_test_size should be 1")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.tp_pairs = tp_pairs
        self.tn_pairs = tn_pairs
        self.train_val_test_size = train_val_test_size
        self.worker = 1 #multiprocessing.cpu_count()

        super(ProcessedDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'map_files.pkl']

    def download(self):
        pass
    
    @staticmethod
    def _encode_onehot(labels):
        ulabels = set(labels)
        ulabels_dict = {c: list(np.identity(len(ulabels))[i, :]) for i, c in enumerate(ulabels)}
        return (np.array(list(map(ulabels_dict.get, labels)), dtype=np.int32), ulabels_dict)
    
    @staticmethod
    def _split_data(tp, tn, shuffle=True, batch_size=512):
        tp['y'] = 1
        tn['y'] = 0
        tp_num = math.ceil((tp.shape[0]/(tp.shape[0]+tn.shape[0]))*batch_size)
        tn_num = math.floor((tn.shape[0]/(tp.shape[0]+tn.shape[0]))*batch_size)
        if shuffle==True:
            tp = tp.sample(frac = 1)
            tn = tn.sample(frac = 1)
        tp_batch = [list(tp.index)[x:x+tp_num] for x in range(0, len(tp.index), tp_num)]
        tn_batch = [list(tn.index)[x:x+tn_num] for x in range(0, len(tn.index), tn_num)]
        if len(tp_batch) == len(tn_batch):
            pass
        elif len(tp_batch) > len(tn_batch):
            tn_batch += [[]]
        else:
            tp_batch += [[]]
        batch = [pd.concat([tp.loc[tp_batch[i],],tn.loc[tn_batch[i],]],axis=0).sample(frac=1).reset_index().drop(columns=['index']) for i in range(len(tp_batch))]
        return batch

    def process(self):
        all_nodes = set()
        all_nodes.update(set(self.raw_edges.source))
        all_nodes.update(set(self.raw_edges.target))
        node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
        idx_map = {j: i for i, j in enumerate(all_nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)
        
        category_array, category_map = self._encode_onehot(node_info.category)
        features = torch.tensor(category_array, dtype=torch.float32)
        map_id = torch.tensor(list(map(idx_map.get, list(node_info.id))), dtype=torch.int32)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=features, edge_index=edge_index, map_id=map_id)
        map_files = [idx_map, category_map]
        
        tp_pairs_train, tp_pairs_val_test = train_test_split(self.tp_pairs,train_size=self.train_val_test_size[0])
        tp_pairs_val, tp_pairs_test = train_test_split(tp_pairs_val_test,train_size=self.train_val_test_size[1]/(self.train_val_test_size[1]+self.train_val_test_size[2]))
        tn_pairs_train, tn_pairs_val_test = train_test_split(self.tn_pairs,train_size=self.train_val_test_size[0])
        tn_pairs_val, tn_pairs_test = train_test_split(tn_pairs_val_test,train_size=self.train_val_test_size[1]/(self.train_val_test_size[1]+self.train_val_test_size[2]))
        
        tp_pairs_train = tp_pairs_train.reset_index().drop(columns=['index'])
        # tp_pairs_train['y'] = 1
        tn_pairs_train = tn_pairs_train.reset_index().drop(columns=['index'])
        # tn_pairs_train['y'] = 0
        # pairs_train = pd.concat([tp_pairs_train,tn_pairs_train], axis=0).sample(frac=1).reset_index().drop(columns=['index'])
        tp_pairs_val = tp_pairs_val.reset_index().drop(columns=['index'])
        # tp_pairs_val['y'] = 1
        tn_pairs_val = tn_pairs_val.reset_index().drop(columns=['index'])
        # tn_pairs_val['y'] = 0
        # pairs_val = pd.concat([tp_pairs_val,tn_pairs_val], axis=0).sample(frac=1).reset_index().drop(columns=['index'])
        tp_pairs_test = tp_pairs_test.reset_index().drop(columns=['index'])
        # tp_pairs_test['y'] = 1
        tn_pairs_test = tn_pairs_test.reset_index().drop(columns=['index'])
        # tn_pairs_test['y'] = 0
        # pairs_test = pd.concat([tp_pairs_test,tn_pairs_test], axis=0).sample(frac=1).reset_index().drop(columns=['index'])

        os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        train_batch = self._split_data(tp=tp_pairs_train, tn=tn_pairs_train, batch_size=self.batch_size)
        # train_loader = []
        for i in trange(len(train_batch)):
            batch_data = train_batch[i]
            train_set = set()
            train_set.update(set(batch_data.source))
            train_set.update(set(batch_data.target))
            train_idx=torch.tensor(list(map(idx_map.get, train_set)), dtype=torch.int32)
            # train_loader += [(n_id, adjs) for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[-1, -1, -1], batch_size=len(train_idx), shuffle=True, num_workers=self.worker)]
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[-1, -1, -1], batch_size=len(train_idx), shuffle=True, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                train_loader = (n_id, adjs)
            filename = 'train_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                pickle.dump(train_loader, output)

        # train_set = set()
        # train_set.update(set(pairs_train.source))
        # train_set.update(set(pairs_train.target))
        # train_idx=torch.tensor(list(map(idx_map.get, train_set)), dtype=torch.long)
        # train_loader = [NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[-1, -1, -1], batch_size=len(train_idx), shuffle=False, num_workers=self.worker)]

        os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        val_batch = self._split_data(tp=tp_pairs_val, tn=tn_pairs_val, batch_size=self.batch_size)
        # val_loader = []
        for i in trange(len(val_batch)):
            batch_data = val_batch[i]
            val_set = set()
            val_set.update(set(batch_data.source))
            val_set.update(set(batch_data.target))
            val_idx=torch.tensor(list(map(idx_map.get, val_set)), dtype=torch.int32)
            # val_loader += [(n_id, adjs) for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=val_idx, sizes=[-1, -1, -1], batch_size=len(val_idx), shuffle=True, num_workers=self.worker)]
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=val_idx, sizes=[-1, -1, -1], batch_size=len(val_idx), shuffle=True, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                val_loader = (n_id, adjs)
            filename = 'val_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(val_loader, output)

        # val_set = set()
        # val_set.update(set(pairs_val.source))
        # val_set.update(set(pairs_val.target))
        # val_idx=torch.tensor(list(map(idx_map.get, val_set)), dtype=torch.long)
        # val_loader = [NeighborSampler(data.edge_index, node_idx=val_idx, sizes=[-1, -1, -1], batch_size=len(val_idx), shuffle=False, num_workers=self.worker)]

        os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        test_batch = self._split_data(tp=tp_pairs_test, tn=tn_pairs_test, batch_size=self.batch_size)
        # test_loader = []
        for i in trange(len(test_batch)):
            batch_data = test_batch[i]
            test_set = set()
            test_set.update(set(batch_data.source))
            test_set.update(set(batch_data.target))
            test_idx=torch.tensor(list(map(idx_map.get, test_set)), dtype=torch.int32)
            # test_loader += [(n_id, adjs) for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=test_idx, sizes=[-1, -1, -1], batch_size=len(test_idx), shuffle=True, num_workers=self.worker)]
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=test_idx, sizes=[-1, -1, -1], batch_size=len(test_idx), shuffle=True, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                test_loader = (n_id, adjs)
            filename = 'test_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                pickle.dump(test_loader, output)


        # test_set = set()
        # test_set.update(set(pairs_test.source))
        # test_set.update(set(pairs_test.target))
        # test_idx=torch.tensor(list(map(idx_map.get, test_set)), dtype=torch.long)
        # test_loader = [NeighborSampler(data.edge_index, node_idx=test_idx, sizes=[-1, -1, -1], batch_size=len(test_idx), shuffle=False, num_workers=self.worker)]

        train_val_test = [train_batch, val_batch, test_batch]
        
        with open(os.path.join(self.processed_dir, 'train_val_test.pkl'), 'wb') as output:
            pickle.dump(train_val_test, output)
        
        # os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        # for index in range(len(train_loader)):
        #     filename = 'train_loader' + '_' + str(index) + '.pkl'
        #     with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
        #         pickle.dump(train_loader[index], output)

        # os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        # for index in range(len(val_loader)):
        #     filename = 'val_loader' + '_' + str(index) + '.pkl'
        #     with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
        #         pickle.dump(val_loader[index], output)

        # os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        # for index in range(len(test_loader)):
        #     filename = 'test_loader' + '_' + str(index) + '.pkl'
        #     with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
        #         pickle.dump(test_loader[index], output)

        with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
            pickle.dump(map_files, output)
        
        torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
    def get_dataset(self):
        data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
        return data
    
    def get_train_val_test(self):
        train_val_test = pickle.load(open(os.path.join(self.processed_dir, 'train_val_test.pkl'), 'rb'))
        return train_val_test
    
    def get_mapfiles(self):
        mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
        return mapfiles
    
    def get_train_loaders_path(self):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return train_loaders_path

    def get_val_loaders_path(self):
        val_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return val_loaders_path

    def get_test_loaders_path(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return test_loaders_path

    # def get_train_loader(self):
    #     train_loader = pickle.load(open(os.path.join(self.processed_dir, 'train_loader.pkl'), 'rb'))
    #     return train_loader

    # def get_val_loader(self):
    #     val_loader = pickle.load(open(os.path.join(self.processed_dir, 'val_loader.pkl'), 'rb'))
    #     return val_loader

    # def get_test_loader(self):
    #     test_loader = pickle.load(open(os.path.join(self.processed_dir, 'test_loader.pkl'), 'rb'))
    #     return test_loader
