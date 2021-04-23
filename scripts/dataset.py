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
    def __init__(self, root, raw_edges, node_info, treat_pairs, not_treat_pairs, contraindicated_for_pairs, transform=None, pre_transform=None, train_val_test_size=[0.8, 0.1, 0.1], batch_size=512, layers=3, dim=100):
        try:
            assert sum(train_val_test_size)==1
        except AssertionError:
            print("The sum of percents in train_val_test_size should be 1")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.treat_pairs = treat_pairs
        self.not_treat_pairs = not_treat_pairs
        self.contraindicated_for_pairs = contraindicated_for_pairs
        self.dim = dim
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
    def _generate_init_emb(idx_map, node_info, dim=100):
        init_embs = dict()
        ulabels = set(node_info.category)
        for label in ulabels:
            curie_ids = node_info.loc[node_info.category.isin([label]),'id']
            curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
            init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
            init_embs[label] = (init_emb, curie_ids)
        return init_embs
    
    @staticmethod
    def _split_data(class_data, shuffle=True, batch_size=512):
        
        for index in range(len(class_data)):
            class_data[index]['y'] = index

        num_list = []
        for index in range(len(class_data)-1):
            num_list.append(math.ceil((class_data[index].shape[0]/pd.concat(class_data).shape[0])*batch_size))
        num_list.append(batch_size-sum(num_list))
        
        if shuffle==True:
            for index in range(len(class_data)):
                class_data[index] = class_data[index].sample(frac = 1)

        batch = []
        count = [0]*len(class_data)
        max_iter = max([math.ceil(class_data[index].shape[0]/num_list[index]) for index in range(len(class_data))])
        for _ in range(max_iter):
            prev = count
            count = [count[index1]+num for index1, num in enumerate(num_list)]
            batch += [pd.concat([class_data[index2].iloc[prev:count[index2],:] for index2 in range(len(class_data))],axis=0).sample(frac=1).reset_index().drop(columns=['index'])]
                
        return batch

    def process(self):
        all_nodes = set()
        all_nodes.update(set(self.raw_edges.source))
        all_nodes.update(set(self.raw_edges.target))
        node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
        idx_map = {j: i for i, j in enumerate(all_nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)
        
        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]

        
        random_state1 = np.random.RandomState(int(time.time()))
        self.treat_pairs['y'] = 0
        self.not_treat_pairs['y'] = 1
        self.contraindicated_for_pairs['y'] = 2
        all_pairs = pd.concat([self.treat_pairs,self.not_treat_pairs,self.contraindicated_for_pairs]).reset_index(drop=True)
        train_index, val_test_index = train_test_split(np.array(list(all_pairs.index)), train_size=self.train_val_test_size[0], random_state=random_state1, shuffle=True, stratify=np.array(list(all_pairs['y'])))
        
        train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
        val_test_pairs = all_pairs.loc[list(val_test_index),:].reset_index(drop=True)

        val_index, test_index = train_test_split(np.array(list(val_test_pairs.index)), train_size=self.train_val_test_size[1]/(self.train_val_test_size[1]+self.train_val_test_size[2]), random_state=random_state1, shuffle=True, stratify=np.array(list(val_test_pairs['y'])))
        
        val_pairs = val_test_pairs.loc[list(val_index),:].reset_index(drop=True)
        test_pairs = val_test_pairs.loc[list(test_index),:].reset_index(drop=True)
        
        
        N = train_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
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
        

        N = val_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        val_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
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
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                

        N = test_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
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
    def __init__(self, root, raw_edges, node_info,  treat_pairs, not_treat_pairs, contraindicated_for_pairs, transform=None, pre_transform=None, K=10, batch_size=512, layers=3, dim=100):
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.treat_pairs = treat_pairs
        self.not_treat_pairs = not_treat_pairs
        self.contraindicated_for_pairs = contraindicated_for_pairs
        self.dim = dim
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
    def _generate_init_emb(idx_map, node_info, dim=100):
        init_embs = dict()
        ulabels = set(node_info.category)
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

        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]

        # seeds random state from time
        random_state1 = np.random.RandomState(int(time.time()))
        # Sets up 10-fold cross validation set
        cv1 = ms.StratifiedKFold(n_splits=self.K, random_state=random_state1, shuffle=True)
        self.treat_pairs['y'] = 0
        self.not_treat_pairs['y'] = 1
        self.contraindicated_for_pairs['y'] = 2
        all_pairs = pd.concat([self.treat_pairs,self.not_treat_pairs,self.contraindicated_for_pairs]).reset_index(drop=True)
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
    def __init__(self, root, raw_edges, node_info, treat_pairs, not_treat_pairs, contraindicated_for_pairs, N=10000, transform=None, pre_transform=None, batch_size=512, layers=3, dim=100):
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.treat_pairs = treat_pairs
        self.not_treat_pairs = not_treat_pairs
        self.contraindicated_for_pairs = contraindicated_for_pairs
        self.dim = dim
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
    def _generate_init_emb(idx_map, node_info, dim=100):
        init_embs = dict()
        ulabels = set(node_info.category)
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

        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim)
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]
        
        # seeds random state from time
        random_state1 = np.random.RandomState(int(time.time()))
        self.treat_pairs['y'] = 0
        self.not_treat_pairs['y'] = 1
        self.contraindicated_for_pairs['y'] = 2
        all_pairs = pd.concat([self.treat_pairs,self.not_treat_pairs,self.contraindicated_for_pairs]).reset_index(drop=True)
        train_index, val_index = train_test_split(np.array(list(all_pairs.index)), train_size=0.9, random_state=random_state1, shuffle=True, stratify=np.array(list(all_pairs['y'])))
        
        train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
        val_pairs = all_pairs.loc[list(val_index),:].reset_index(drop=True)
        
        
        N = train_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
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
        
        
        N = val_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)  

        
        val_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
            val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(val_batch)):

            batch_data = test_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'val_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
        
        # generate random pairs of drug and disease
        disease_list = [node_id for node_id, node_type in id_to_type.items() if node_type=='disease' or node_type=='phenotypic_feature']
        drug_list = [node_id for node_id, node_type in id_to_type.items() if node_type=='chemical_substance']
        random_pairs = self._rand_rate(self.N, drug_list, disease_list, idx_map)
        
        N = self.treat_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        treat_pairs_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'treat_pairs_loaders'))
        for _, index in cv2.split(np.array(list(self.treat_pairs.index)), [0]*self.treat_pairs.shape[0]):
            treat_pairs_batch += [self.treat_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(treat_pairs_batch)):

            batch_data = treat_pairs_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'treat_pairs_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'treat_pairs_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                
        N = self.not_treat_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        not_treat_pairs_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'not_treat_pairs_loaders'))
        for _, index in cv2.split(np.array(list(self.not_treat_pairs.index)), [1]*self.not_treat_pairs.shape[0]):
            not_treat_pairs_batch += [self.not_treat_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(not_treat_pairs_batch)):

            batch_data = not_treat_pairs_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'not_treat_pairs_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'not_treat_pairs_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)

        N = self.contraindicated_for_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

        contraindicated_for_pairs_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'contraindicated_for_pairs_loaders'))
        for _, index in cv2.split(np.array(list(self.contraindicated_for_pairs.index)), [1]*self.contraindicated_for_pairs.shape[0]):
            contraindicated_for_pairs_batch += [self.contraindicated_for_pairs.loc[list(index),:].reset_index(drop=True)]
        for i in trange(len(contraindicated_for_pairs_batch)):

            batch_data = contraindicated_for_pairs_batch[i]
            data_set = set()
            data_set.update(set(batch_data.source))
            data_set.update(set(batch_data.target))
            data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
            for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
                adjs = [(adj.edge_index,adj.size) for adj in adjs]
                loader = (n_id, adjs)
            filename = 'contraindicated_for_pairs_loader' + '_' + str(i) + '.pkl'
            with open(os.path.join(self.processed_dir, 'contraindicated_for_pairs_loaders', filename), 'wb') as output:
                pickle.dump(loader, output)
                
        N = random_pairs.shape[0]//self.batch_size
        # seeds random state from time
        random_state2 = np.random.RandomState(int(time.time()))
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
                
        batch_set = [train_batch, val_batch, treat_pairs_batch, not_treat_pairs_batch, contraindicated_for_pairs_batch, rp_batch]
        
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
    
    def get_treat_pairs_loader(self):
        treat_pairs_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'treat_pairs_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(treat_pairs_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_not_treat_pairs_loader(self):
        not_treat_pairs_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'not_treat_pairs_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(not_treat_pairs_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

    def get_contraindicated_for_pairs_loader(self):
        contraindicated_for_pairs_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'contraindicated_for_pairs_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(contraindicated_for_pairs_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    def get_rp_loader(self):
        rp_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'rp_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
        return DataLoader(DataWrapper(rp_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)