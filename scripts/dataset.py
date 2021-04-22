import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data, NeighborSampler
import pickle
import math
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
import multiprocessing
from glob import glob
from utils import DataWrapper
import time
import random

class ProcessedDataset(InMemoryDataset):
    def __init__(self, root, raw_edges, node_info, tp_pairs, tn_pairs, transform=None, pre_transform=None, train_val_test_size=[0.8, 0.1, 0.1], batch_size=512, layers=3, dim=100, known_int_emb_dict=None):
        if not sum(train_val_test_size)==1:
            raise AssertionError("The sum of percents in train_val_test_size should be 1")
        if known_int_emb_dict is not None:
            if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
                raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.tp_pairs = tp_pairs
        self.tn_pairs = tn_pairs
        self.dim = dim
        self.known_int_emb_dict = known_int_emb_dict
        self.train_val_test_size = train_val_test_size
        self.worker = 4 #multiprocessing.cpu_count()
        self.layer_size = []
        for _ in range(layers):
            self.layer_size += [-1]

        super(ProcessedDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'map_files.pkl', 'train_val_test.pkl']

    def download(self):
        pass
    
    @staticmethod
    def _encode_onehot(labels):
        ulabels = set(labels)
        ulabels_dict = {c: list(np.identity(len(ulabels))[i, :]) for i, c in enumerate(ulabels)}
        return (np.array(list(map(ulabels_dict.get, labels)), dtype=np.int32), ulabels_dict)

    @staticmethod
    def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
        init_embs = dict()
        ulabels = list(set(node_info.category))
        if known_int_emb_dict is not None:
            known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
            known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
            category_has_known_init_emb = set(known_int_emb_df['category'])
            for category in category_has_known_init_emb:
                try:
                    assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
                except AssertionError:
                    print(f"Not all curies with cateogry {category} have known intial embedding")    
                curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
                curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
                init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])), dtype=torch.float32)
                init_embs[category] = (init_emb, curie_ids)
                ulabels.remove(category)
                
        for label in ulabels:
            curie_ids = node_info.loc[node_info.category.isin([label]),'id']
            curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
            init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
            init_embs[label] = (init_emb, curie_ids)
        return init_embs
    
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
        
#         category_array, category_map = self._encode_onehot(node_info.category)
        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict=self.known_int_emb_dict)
#         features = torch.tensor(category_array, dtype=torch.float32)
#         map_id = torch.tensor(list(map(idx_map.get, list(node_info.id))), dtype=torch.int32)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
#         data = Data(feat=features, edge_index=edge_index, map_id=map_id)
#         x = torch.vstack([init_embs[key][0] for key in init_embs])
#         indexes = torch.hstack([init_embs[key][1] for key in init_embs])
#         x = x[indexes.sort().indices]
        data = Data(feat=init_embs, edge_index=edge_index)
#         map_files = [idx_map, category_map]
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]
        
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

        temp_batch = []
        os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        train_batch = self._split_data(tp=tp_pairs_train, tn=tn_pairs_train, batch_size=self.batch_size)
        for i in trange(len(train_batch)):

            ## skip this batch if there is only one class in this batch
            if len(set(train_batch[i]['y'])) == 1:
                temp_batch += [train_batch[i]]
                del train_batch[i]
                continue
            batch_data = train_batch[i]
            train_set = set()
            train_set.update(set(batch_data.source))
            train_set.update(set(batch_data.target))
            train_idx=torch.tensor(list(map(idx_map.get, train_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=train_idx, sizes=self.layer_size, batch_size=len(train_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                train_loader = (n_id, adjs)
            filename = 'train_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                pickle.dump(train_loader, output)

        os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        val_batch = self._split_data(tp=tp_pairs_val, tn=tn_pairs_val, batch_size=self.batch_size)
        for i in trange(len(val_batch)):

            ## skip this batch if there is only one class in this batch
            if len(set(val_batch[i]['y'])) == 1:
                temp_batch += [val_batch[i]]
                del val_batch[i]
                continue
            batch_data = val_batch[i]
            val_set = set()
            val_set.update(set(batch_data.source))
            val_set.update(set(batch_data.target))
            val_idx=torch.tensor(list(map(idx_map.get, val_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=val_idx, sizes=self.layer_size, batch_size=len(val_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                val_loader = (n_id, adjs)
            filename = 'val_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(val_loader, output)

        os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        test_batch = self._split_data(tp=tp_pairs_test, tn=tn_pairs_test, batch_size=self.batch_size)
        test_batch += temp_batch ## put the inblanced train/val batch to test batch list
        for i in trange(len(test_batch)):

            # ## skip this batch if there is only one class in this batch
            # if len(set(test_batch[i]['y'])) == 1:
            #     del test_batch[i]
            #     continue
            batch_data = test_batch[i]
            test_set = set()
            test_set.update(set(batch_data.source))
            test_set.update(set(batch_data.target))
            test_idx=torch.tensor(list(map(idx_map.get, test_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=test_idx, sizes=self.layer_size, batch_size=len(test_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                test_loader = (n_id, adjs)
            filename = 'test_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                pickle.dump(test_loader, output)

        train_val_test = [train_batch, val_batch, test_batch]
        
        with open(os.path.join(self.processed_dir, 'train_val_test.pkl'), 'wb') as output:
            pickle.dump(train_val_test, output)

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

    def get_train_loader(self):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_val_loader(self):
        val_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_test_loader(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)


class MakeKFoldData(InMemoryDataset):
    def __init__(self, root, raw_edges, node_info, tp_pairs, tn_pairs, transform=None, pre_transform=None, K=10, batch_size=512, layers=3, dim=100, known_int_emb_dict=None):
        if known_int_emb_dict is not None:
            if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
                raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.tp_pairs = tp_pairs
        self.tn_pairs = tn_pairs
        self.dim = dim
        self.known_int_emb_dict = known_int_emb_dict
        self.K = K
        self.worker = 4 #multiprocessing.cpu_count()
        self.layer_size = []
        for _ in range(layers):
            self.layer_size += [-1]
            
        super(MakeKFoldData, self).__init__(root, transform, pre_transform)
            
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'map_files.pkl']

    def download(self):
        pass
    
    @staticmethod
    def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
        init_embs = dict()
        ulabels = list(set(node_info.category))
        if known_int_emb_dict is not None:
            known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
            known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
            category_has_known_init_emb = set(known_int_emb_df['category'])
            for category in category_has_known_init_emb:
                try:
                    assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
                except AssertionError:
                    print(f"Not all curies with cateogry {category} have known intial embedding")    
                curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
                curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
                init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])), dtype=torch.float32)
                init_embs[category] = (init_emb, curie_ids)
                ulabels.remove(category)
                
        for label in ulabels:
            curie_ids = node_info.loc[node_info.category.isin([label]),'id']
            curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
            init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
            init_embs[label] = (init_emb, curie_ids)
        return init_embs
    
    def process(self):
        all_nodes = set()
        all_nodes.update(set(self.raw_edges.source))
        all_nodes.update(set(self.raw_edges.target))
        node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
        idx_map = {j: i for i, j in enumerate(all_nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)

        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict=self.known_int_emb_dict)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]

        # seeds random state from time
        random_state1 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv1 = ms.StratifiedKFold(n_splits=self.K, random_state=random_state1, shuffle=True)
        self.tp_pairs['y'] = 1
        self.tn_pairs['y'] = 0
        all_pairs = pd.concat([self.tp_pairs,self.tn_pairs]).reset_index(drop=True)
        for fold, (train_index, test_index) in enumerate(cv1.split(np.array(list(all_pairs.index)), np.array(all_pairs['y']))):
            train_index, val_index = train_test_split(train_index, test_size=1/9, random_state=random_state1, shuffle=True, stratify=np.array(all_pairs.loc[list(train_index),'y']))
            train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
            val_pairs = all_pairs.loc[list(val_index),:].reset_index(drop=True)
            test_pairs = all_pairs.loc[list(test_index),:].reset_index(drop=True)
              
            os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}"))
            N = train_pairs.shape[0]//self.batch_size
            # seeds random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            # Sets up 10-fold cross validation set
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            
            train_batch = list()
            os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_loaders'))
            for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
                train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]
            for i in trange(len(train_batch)):

                batch_data = train_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'train_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)

            N = val_pairs.shape[0]//self.batch_size
            # seeds random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            # Sets up 10-fold cross validation set
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            
            val_batch = list()
            os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'val_loaders'))
            for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
                val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]
            for i in trange(len(val_batch)):

                batch_data = val_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'val_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'val_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)                    
                    
            N = test_pairs.shape[0]//self.batch_size
            # seeds random state from time
            random_state2 = np.random.RandomState(int(time.time()))
            # Sets up 10-fold cross validation set
            cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)                    
                    
            test_batch = list()
            os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'test_loaders'))
            for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
                test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]
            for i in trange(len(test_batch)):

                batch_data = test_batch[i]
                data_set = set()
                data_set.update(set(batch_data.source))
                data_set.update(set(batch_data.target))
                data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
                for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                    adjs = [(adj.edge_index,adj.size) for adj in adjs]
                    loader = (n_id, adjs)
                filename = 'test_loader' + '_' + str(i) + '.pkl'
                with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'test_loaders', filename), 'wb') as output:
                    pickle.dump(loader, output)

            train_val_test = [train_batch, val_batch, test_batch]

            with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_val_test.pkl'), 'wb') as output:
                pickle.dump(train_val_test, output)

        with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
            pickle.dump(map_files, output)
        
        torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
        
    def get_dataset(self):
        data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
        return data

    def get_mapfiles(self):
        mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
        return mapfiles
    
    def get_train_val_test(self, fold):
        train_val_test = pickle.load(open(os.path.join(self.processed_dir, f"fold{fold}", 'train_val_test.pkl'), 'rb'))
        return train_val_test

    def get_train_loader(self, fold):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_val_loader(self, fold):
        val_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_test_loader(self, fold):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    

class MakeKRandomPairs(InMemoryDataset):
    def __init__(self, root, raw_edges, node_info, tp_pairs, tn_pairs, N=10000, transform=None, pre_transform=None, batch_size=512, layers=3, dim=100, known_int_emb_dict=None):
        if known_int_emb_dict is not None:
            if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
                raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.tp_pairs = tp_pairs
        self.tn_pairs = tn_pairs
        self.dim = dim
        self.known_int_emb_dict = known_int_emb_dict
        self.N = N
        self.worker = 4 #multiprocessing.cpu_count()
        self.layer_size = []
        for _ in range(layers):
            self.layer_size += [-1]
            
        super(MakeKRandomPairs, self).__init__(root, transform, pre_transform)
            
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_kg.dataset', 'map_files.pkl']

    def download(self):
        pass
    
    @staticmethod
    def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
        init_embs = dict()
        ulabels = list(set(node_info.category))
        if known_int_emb_dict is not None:
            known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
            known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
            category_has_known_init_emb = set(known_int_emb_df['category'])
            for category in category_has_known_init_emb:
                try:
                    assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
                except AssertionError:
                    print(f"Not all curies with cateogry {category} have known intial embedding")    
                curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
                curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
                init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])), dtype=torch.float32)
                init_embs[category] = (init_emb, curie_ids)
                ulabels.remove(category)
                
        for label in ulabels:
            curie_ids = node_info.loc[node_info.category.isin([label]),'id']
            curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
            init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
            init_embs[label] = (init_emb, curie_ids)
        return init_embs
    
    @staticmethod
    def _rand_rate(n, drug_list, disease_list, idx_map):

        random.seed(int(time.time()/100))
        idtoname = {value:key for key, value in idx_map.items()}

        # get number of drug and disease ids
        drug_n = len(drug_list)
        dis_n = len(disease_list)

        # Find all permutations
        perms = zip(random.choices(list(range(drug_n)),k=2*n), random.choices(list(range(dis_n)),k=2*n))
        random_pairs = pd.DataFrame([(idtoname[drug_list[idx[0]]], idtoname[disease_list[idx[1]]]) for idx in list(perms)])
        random_pairs = random_pairs.rename(columns={0:'source', 1:'target'})
        random_pairs['y'] = 1
        
        print(f'Number of random pairs: {random_pairs.shape[0]}\n')

        return random_pairs
    
    
    def process(self):
        all_nodes = set()
        all_nodes.update(set(self.raw_edges.source))
        all_nodes.update(set(self.raw_edges.target))
        node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
        idx_map = {j: i for i, j in enumerate(all_nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)

        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict = self.known_int_emb_dict)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]
        
        # seeds random state from time
        random_state1 = np.random.RandomState(int(time.time()))
        self.tp_pairs['y'] = 1
        self.tn_pairs['y'] = 0
        all_pairs = pd.concat([self.tp_pairs,self.tn_pairs]).reset_index(drop=True)
        train_index, test_index = train_test_split(np.array(list(all_pairs.index)), train_size=0.9, random_state=random_state1, shuffle=True, stratify=np.array(list(all_pairs['y'])))
        
        train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
        test_pairs = all_pairs.loc[list(test_index),:].reset_index(drop=True)
        
        
        N = train_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        train_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
            train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(train_batch)):

            batch_data = train_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'train_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
        
        
        N = test_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)  

        
        test_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
            test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(test_batch)):

            batch_data = test_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'test_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'test_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
        
        # generate random pairs of drug and disease
        disease_list = [node_id for node_id, node_type in id_to_type.items() if node_type=='disease' or node_type=='phenotypic_feature']
        drug_list = [node_id for node_id, node_type in id_to_type.items() if node_type=='chemical_substance']
        random_pairs = self._rand_rate(self.N, drug_list, disease_list, idx_map)
        
        N = self.tp_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        tp_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'tp_loaders'))
        for _, index in cv2.split(np.array(list(self.tp_pairs.index)), [1]*self.tp_pairs.shape[0]):
            tp_batch += [self.tp_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(tp_batch)):

            batch_data = tp_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'tp_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'tp_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                
        N = self.tn_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        tn_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'tn_loaders'))
        for _, index in cv2.split(np.array(list(self.tn_pairs.index)), [1]*self.tn_pairs.shape[0]):
            tn_batch += [self.tn_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(tn_batch)):

            batch_data = tn_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'tn_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'tn_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                       
        N = random_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        rp_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'rp_loaders'))
        for _, index in cv2.split(np.array(list(random_pairs.index)), [1]*random_pairs.shape[0]):
            rp_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(rp_batch)):

            batch_data = rp_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'rp_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'rp_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                
        batch_set = [train_batch, test_batch, tp_batch, tn_batch, rp_batch]
        
        with open(os.path.join(self.processed_dir, 'batch_set.pkl'), 'wb') as output:
            pickle.dump(batch_set, output)

        with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
            pickle.dump(map_files, output)
        
        torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
    def get_dataset(self):
        data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
        return data

    def get_mapfiles(self):
        mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
        return mapfiles
    
    def get_batch_set(self):
        batch_set = pickle.load(open(os.path.join(self.processed_dir, 'batch_set.pkl'), 'rb'))
        return batch_set

    def get_train_loader(self):
        train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_test_loader(self):
        test_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_tp_loader(self):
        tp_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'tp_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(tp_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_tn_loader(self):
        tn_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'tn_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(tn_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_rp_loader(self):
        rp_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'rp_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(rp_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)