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
    def __init__(self, root, run_model, raw_edges, node_info, tp_pairs, tn_pairs, all_known_tp_pairs, transform=None, pre_transform=None, train_val_test_size=[0.8, 0.1, 0.1], batch_size=512, layers=3, dim=100, known_int_emb_dict=None, train_N=30, non_train_N=500, num_samples=None, seed=1234):
        if not sum(train_val_test_size)==1:
            raise AssertionError("The sum of percents in train_val_test_size should be 1")
#         if known_int_emb_dict is not None:
#             if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
#                 raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
        self.raw_edges = raw_edges[['source','target']]
        self.node_info = node_info
        self.batch_size = batch_size
        self.tp_pairs = tp_pairs
        self.tn_pairs = tn_pairs
        self.all_known_tp_pairs = all_known_tp_pairs
        self.dim = dim
        self.known_int_emb_dict = known_int_emb_dict
        self.train_val_test_size = train_val_test_size
        self.N = N
        self.worker = 4 #multiprocessing.cpu_count()
        self.seed = seed

        self.layer_size = [1000, 1000, 1000, 1000]

        self.layer_size = []
        if run_model == 'gat' and num_samples is None:
            for _ in range(layers):
                self.layer_size += [-1]
        else:
            self.layer_size = num_samples

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
                init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])).astype(float), dtype=torch.float32)
                init_embs[category] = (init_emb, curie_ids)
                ulabels.remove(category)
            
            print(f"Number of categories that are not uninitialized with known embeddings: {len(ulabels)}")

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


    @staticmethod
    def _rand_rate(n, test_pairs, disease_list, idx_map, all_known_tp_pairs):

        random.seed(int(self.seed))
        idtoname = {value:key for key, value in idx_map.items()}
        ## only use the tp data
        test_pairs = test_pairs.loc[test_pairs['y'] == 0,:].reset_index(drop=True)
        drug_in_test_data = list(set(test_pairs['source']))
        disease_name_list = list(map(idtoname.get, disease_list))
        
        ## create a check list for all tp an tn pairs
        check_list_temp = {(all_known_tp_pairs.loc[index,'source'],all_known_tp_pairs.loc[index,'target']):1 for index in range(all_known_tp_pairs.shape[0])}
        
        random_pairs = []
        for drug in drug_in_test_data:
            count = 0
            temp_dict = dict()
            random.shuffle(disease_name_list)
            for disease in disease_name_list:
                if (drug, disease) not in check_list_temp and (drug, disease) not in temp_dict:
                    temp_dict[(drug, disease)] = 1
                    count += 1
                if count == n:
                    break
            random_pairs += [pd.DataFrame(temp_dict.keys())]
        
        random_pairs = pd.concat(random_pairs).reset_index(drop=True).rename(columns={0:'source',1:'target'})
        random_pairs['y'] = 1
        
        print(f'Number of random pairs: {random_pairs.shape[0]}', flush=True)

        return random_pairs
    
    def process(self):
        all_nodes = set()
        all_nodes.update(set(self.raw_edges.source))
        all_nodes.update(set(self.raw_edges.target))
        node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
        idx_map = {j: i for i, j in enumerate(all_nodes)}
        edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)
        
        ## generate initial embedding vectors for each category
        init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict=self.known_int_emb_dict)
        ## generate edge index matrix
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        data = Data(feat=init_embs, edge_index=edge_index)
        id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
        typeid = {key:index for index, key in enumerate(init_embs)}
        map_files = [idx_map, id_to_type, typeid]
        
        ## split dataset to training, validation and test according 
        # seed random state from time
        ### generate train, validation and test pairs
        print("", flush=True)
        print(f"generate train, validation and test pairs", flush=True)
        random_state = np.random.RandomState(int(self.seed))
        self.tp_pairs['y'] = 0
        self.tn_pairs['y'] = 2
        all_pairs = pd.concat([self.tp_pairs,self.tn_pairs]).reset_index(drop=True)
        train_index, val_test_index = train_test_split(np.array(list(all_pairs.index)), train_size=self.train_val_test_size[0], random_state=random_state, shuffle=True, stratify=np.array(list(all_pairs['y'])))        
        train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
        val_test_pairs = all_pairs.loc[list(val_test_index),:].reset_index(drop=True)
        
        val_index, test_index = train_test_split(np.array(list(val_test_pairs.index)), train_size=self.train_val_test_size[1]/(self.train_val_test_size[1]+self.train_val_test_size[2]), random_state=random_state, shuffle=True, stratify=np.array(list(val_test_pairs['y'])))
        val_pairs = val_test_pairs.loc[list(val_index),:].reset_index(drop=True)
        test_pairs = val_test_pairs.loc[list(test_index),:].reset_index(drop=True)
        
        
        ### generate random pairs for MRR or Hit@K evaluation
        print("", flush=True)
        print(f"generate random pairs for MRR or Hit@K evaluation", flush=True)
        disease_list = list(set([node_id for node_id, node_type in id_to_type.items() if node_type=='biolink:Disease' or node_type=='biolink:PhenotypicFeature' or node_type=='biolink:DiseaseOrPhenotypicFeature']))
        train_random_pairs = self._rand_rate(self.train_N, train_pairs, disease_list, idx_map, self.all_known_tp_pairs)
        val_random_pairs = self._rand_rate(self.non_train_N, val_pairs, disease_list, idx_map, self.all_known_tp_pairs)
        test_random_pairs = self._rand_rate(self.non_train_N, test_pairs, disease_list, idx_map, self.all_known_tp_pairs)
        
        ## split training set according to the given batch size
        train_pairs = pd.concat([train_pairs,train_random_pairs]).reset_index(drop=True)
        N = train_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state, shuffle=True)
        train_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
        for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
            train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]
        print("", flush=True)
        print(f"generating batches for train set", flush=True)
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
        
        
        ## split validation set according to the given batch size
        val_pairs = pd.concat([val_pairs,val_random_pairs]).reset_index(drop=True)
        N = val_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state, shuffle=True)
        val_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'val_loaders'))
        for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
            val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]
        print("", flush=True)
        print(f"generating batches for validation set", flush=True)
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
            
        
        ## split test set according to the given batch size
        test_pairs = pd.concat([test_pairs,test_random_pairs]).reset_index(drop=True)
        N = test_pairs.shape[0]//self.batch_size
        # Sets up 10-fold cross validation set
        cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state, shuffle=True)    
        test_batch = list()
        os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
        print("", flush=True)
        print(f"generating batches for test set", flush=True)
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

                
        # ## split random pair set according to the given batch size
        # N = random_pairs.shape[0]//self.batch_size
        # # seed random state from time
        # random_state2 = np.random.RandomState(int(time.time()))
        # # Sets up 10-fold cross validation set
        # cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)    
        # random_batch = list()
        # os.mkdir(os.path.join(self.processed_dir, 'random_loaders'))
        # print("", flush=True)
        # print(f"generating batches for random pair set", flush=True)
        # for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
        #     random_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]
        # for i in trange(len(random_batch)):

        #     batch_data = random_batch[i]
        #     data_set = set()
        #     data_set.update(set(batch_data.source))
        #     data_set.update(set(batch_data.target))
        #     data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
        #     for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
        #         adjs = [(adj.edge_index,adj.size) for adj in adjs]
        #         loader = (n_id, adjs)
        #     filename = 'random_loader' + '_' + str(i) + '.pkl'
        #     with open(os.path.join(self.processed_dir, 'random_loaders', filename), 'wb') as output:
        #         pickle.dump(loader, output)

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

    # def get_random_loader(self):
    #     random_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'random_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
    #     return DataLoader(DataWrapper(random_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)



# class MakeKFoldData(InMemoryDataset):
#     def __init__(self, root, run_model, raw_edges, node_info, tp_pairs, tn_pairs, all_known_tp_pairs, transform=None, pre_transform=None, K=10, batch_size=512, layers=3, dim=100, known_int_emb_dict=None, N=500, num_samples=None):
# #         if known_int_emb_dict is not None:
# #             if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
# #                 raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
#         self.raw_edges = raw_edges[['source','target']]
#         self.node_info = node_info
#         self.batch_size = batch_size
#         self.tp_pairs = tp_pairs
#         self.tn_pairs = tn_pairs
#         self.all_known_tp_pairs = all_known_tp_pairs
#         self.dim = dim
#         self.known_int_emb_dict = known_int_emb_dict
#         self.K = K
#         self.N = N
#         self.worker = 4 #multiprocessing.cpu_count()
#         self.layer_size = []
#         if run_model == 'gat':
#             for _ in range(layers):
#                 self.layer_size += [-1]
#         else:
#             self.layer_size = num_samples

#         super(MakeKFoldData, self).__init__(root, transform, pre_transform)
            
#     @property
#     def raw_file_names(self):
#         return []
#     @property
#     def processed_file_names(self):
#         return ['processed_kg.dataset', 'map_files.pkl']

#     def download(self):
#         pass
    
#     @staticmethod
#     def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
#         init_embs = dict()
#         ulabels = list(set(node_info.category))
#         if known_int_emb_dict is not None:
#             known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
#             known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
#             category_has_known_init_emb = set(known_int_emb_df['category'])
#             for category in category_has_known_init_emb:
#                 try:
#                     assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
#                 except AssertionError:
#                     print(f"Not all curies with cateogry {category} have known intial embedding")    
#                 curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
#                 curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
#                 init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])), dtype=torch.float32)
#                 init_embs[category] = (init_emb, curie_ids)
#                 ulabels.remove(category)

#         for label in ulabels:
#             curie_ids = node_info.loc[node_info.category.isin([label]),'id']
#             curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
#             init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
#             init_embs[label] = (init_emb, curie_ids)

#         return init_embs

    
#     @staticmethod
#     def _rand_rate(n, test_pairs, disease_list, idx_map, all_known_tp_pairs):

#         random.seed(int(time.time()/100))
#         idtoname = {value:key for key, value in idx_map.items()}
#         ## only use the tp data
#         test_pairs = test_pairs.loc[test_pairs['y'] == 1,:].reset_index(drop=True)
#         drug_in_test_data = list(set(test_pairs['source']))
#         disease_name_list = list(map(idtoname.get, disease_list))
        
#         ## create a check list for all tp an tn pairs
#         check_list_temp = {(all_known_tp_pairs.loc[index,'source'],all_known_tp_pairs.loc[index,'target']):1 for index in range(all_known_tp_pairs.shape[0])}
        
#         random_pairs = []
#         for drug in drug_in_test_data:
#             count = 0
#             temp_dict = dict()
#             random.shuffle(disease_name_list)
#             for disease in disease_name_list:
#                 if (drug, disease) not in check_list_temp and (drug, disease) not in temp_dict:
#                     temp_dict[(drug, disease)] = 1
#                     count += 1
#                 if count == n:
#                     break
#             random_pairs += [pd.DataFrame(temp_dict.keys())]
        
#         random_pairs = pd.concat(random_pairs).reset_index(drop=True).rename(columns={0:'source',1:'target'})
#         random_pairs['y'] = 1
        
#         print(f'Number of random pairs: {random_pairs.shape[0]}', flush=True)

#         return random_pairs
    
#     def process(self):
#         all_nodes = set()
#         all_nodes.update(set(self.raw_edges.source))
#         all_nodes.update(set(self.raw_edges.target))
#         node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
#         idx_map = {j: i for i, j in enumerate(all_nodes)}
#         edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)

#         init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict=self.known_int_emb_dict)
#         edge_index = torch.tensor(edges.T, dtype=torch.long)
#         data = Data(feat=init_embs, edge_index=edge_index)
#         id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
#         typeid = {key:index for index, key in enumerate(init_embs)}
#         map_files = [idx_map, id_to_type, typeid]

#         # seed random state from time
#         random_state1 = np.random.RandomState(int(time.time()))
#         # Sets up 10-fold cross validation set
#         cv1 = ms.StratifiedKFold(n_splits=self.K, random_state=random_state1, shuffle=True)
#         self.tp_pairs['y'] = 1
#         self.tn_pairs['y'] = 0
#         all_pairs = pd.concat([self.tp_pairs,self.tn_pairs]).reset_index(drop=True)
#         for fold, (train_index, test_index) in enumerate(cv1.split(np.array(list(all_pairs.index)), np.array(all_pairs['y']))):
#             train_index, val_index = train_test_split(train_index, test_size=1/9, random_state=random_state1, shuffle=True, stratify=np.array(all_pairs.loc[list(train_index),'y']))
#             train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
#             val_pairs = all_pairs.loc[list(val_index),:].reset_index(drop=True)
#             test_pairs = all_pairs.loc[list(test_index),:].reset_index(drop=True)
              
# #             ### generate random pairs for MRR or Hit@K evaluation
# #             print("", flush=True)
# #             print(f"generate random pairs for MRR or Hit@K evaluation", flush=True)
# #             disease_list = list(set([node_id for node_id, node_type in id_to_type.items() if node_type=='biolink:Disease' or node_type=='biolink:PhenotypicFeature' or node_type=='biolink:DiseaseOrPhenotypicFeature']))
# #             random_pairs = self._rand_rate(self.N, test_pairs, disease_list, idx_map, self.all_known_tp_pairs)
                
#             os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}"))
#             print("", flush=True)
#             print(f"generating batches for fold{fold+1} data set", flush=True)
#             N = train_pairs.shape[0]//self.batch_size
#             # seed random state from time
#             random_state2 = np.random.RandomState(int(time.time()))
#             # Sets up 10-fold cross validation set
#             cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            
#             train_batch = list()
#             os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_loaders'))
#             for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
#                 train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]
#             print("", flush=True)
#             print(f"generating batches for train set", flush=True)
#             for i in trange(len(train_batch)):

#                 batch_data = train_batch[i]
#                 data_set = set()
#                 data_set.update(set(batch_data.source))
#                 data_set.update(set(batch_data.target))
#                 data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#                 for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                     adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                     loader = (n_id, adjs)
#                 filename = 'train_loader' + '_' + str(i) + '.pkl'
#                 with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_loaders', filename), 'wb') as output:
#                     pickle.dump(loader, output)

#             N = val_pairs.shape[0]//self.batch_size
#             # seed random state from time
#             random_state2 = np.random.RandomState(int(time.time()))
#             cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)
            
#             val_batch = list()
#             os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'val_loaders'))
#             for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
#                 val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]
#             print("", flush=True)
#             print(f"generating batches for validation set", flush=True)
#             for i in trange(len(val_batch)):

#                 batch_data = val_batch[i]
#                 data_set = set()
#                 data_set.update(set(batch_data.source))
#                 data_set.update(set(batch_data.target))
#                 data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#                 for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                     adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                     loader = (n_id, adjs)
#                 filename = 'val_loader' + '_' + str(i) + '.pkl'
#                 with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'val_loaders', filename), 'wb') as output:
#                     pickle.dump(loader, output)                    
                    
#             N = test_pairs.shape[0]//self.batch_size
#             # seed random state from time
#             random_state2 = np.random.RandomState(int(time.time()))
#             cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)                    
                    
#             test_batch = list()
#             os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'test_loaders'))
#             for _, index in cv2.split(np.array(list(test_pairs.index)), np.array(test_pairs['y'])):
#                 test_batch += [test_pairs.loc[list(index),:].reset_index(drop=True)]
#             print("", flush=True)
#             print(f"generating batches for test set", flush=True)
#             for i in trange(len(test_batch)):

#                 batch_data = test_batch[i]
#                 data_set = set()
#                 data_set.update(set(batch_data.source))
#                 data_set.update(set(batch_data.target))
#                 data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#                 for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                     adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                     loader = (n_id, adjs)
#                 filename = 'test_loader' + '_' + str(i) + '.pkl'
#                 with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'test_loaders', filename), 'wb') as output:
#                     pickle.dump(loader, output)

# #             N = random_pairs.shape[0]//self.batch_size
# #             # seed random state from time
# #             random_state2 = np.random.RandomState(int(time.time()))
# #             # Sets up 10-fold cross validation set
# #             cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)   

# #             random_batch = list()
# #             os.mkdir(os.path.join(self.processed_dir, f"fold{fold+1}", 'random_loaders'))
# #             print("", flush=True)
# #             print(f"generating batches for random pair set", flush=True)
# #             for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
# #                 random_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]
# #             for i in trange(len(random_batch)):

# #                 batch_data = random_batch[i]
# #                 data_set = set()
# #                 data_set.update(set(batch_data.source))
# #                 data_set.update(set(batch_data.target))
# #                 data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
# #                 for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
# #                     adjs = [(adj.edge_index,adj.size) for adj in adjs]
# #                     loader = (n_id, adjs)
# #                 filename = 'random_loader' + '_' + str(i) + '.pkl'
# #                 with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'random_loaders', filename), 'wb') as output:
# #                     pickle.dump(loader, output)
                    
#             train_val_test = [train_batch, val_batch, test_batch]
# #             train_val_test_random = [train_batch, val_batch, test_batch, random_batch]

#             with open(os.path.join(self.processed_dir, f"fold{fold+1}", 'train_val_test.pkl'), 'wb') as output:
#                 pickle.dump(train_val_test, output)

#         with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
#             pickle.dump(map_files, output)
        
#         torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
        
#     def get_dataset(self):
#         data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
#         return data

#     def get_mapfiles(self):
#         mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
#         return mapfiles
    
#     def get_train_val_test(self, fold):
#         train_val_test = pickle.load(open(os.path.join(self.processed_dir, f"fold{fold}", 'train_val_test.pkl'), 'rb'))
#         return train_val_test

#     def get_train_loader(self, fold):
#         train_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

#     def get_val_loader(self, fold):
#         val_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
#     def get_test_loader(self, fold):
#         test_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'test_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(test_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
# #     def get_random_loader(self, fold):
# #         random_loaders_path = sorted(glob(os.path.join(self.processed_dir, f"fold{fold}", 'random_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
# #         return DataLoader(DataWrapper(random_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
    
# class MakeKRandomPairs(InMemoryDataset):
#     def __init__(self, root, raw_edges, node_info, tp_pairs, tn_pairs, N=10000, transform=None, pre_transform=None, batch_size=512, layers=3, dim=100, known_int_emb_dict=None):
# #         if known_int_emb_dict is not None:
# #             if not all([len(known_int_emb_dict[key])==dim for key, value in known_int_emb_dict.items()]):
# #                 raise AssertionError(f"At least one known inital embedding is not eqaul to the dimension of intial embedding you set which is {dim}")
#         self.raw_edges = raw_edges[['source','target']]
#         self.node_info = node_info
#         self.batch_size = batch_size
#         self.tp_pairs = tp_pairs
#         self.tn_pairs = tn_pairs
#         self.dim = dim
#         self.known_int_emb_dict = known_int_emb_dict
#         self.N = N
#         self.worker = 1 #multiprocessing.cpu_count()
#         self.layer_size = []
#         for _ in range(layers):
#             self.layer_size += [-1]
            
#         super(MakeKRandomPairs, self).__init__(root, transform, pre_transform)
            
#     @property
#     def raw_file_names(self):
#         return []
#     @property
#     def processed_file_names(self):
#         return ['processed_kg.dataset', 'map_files.pkl']

#     def download(self):
#         pass
    
#     @staticmethod
#     def _generate_init_emb(idx_map, node_info, dim=100, known_int_emb_dict=None):
#         init_embs = dict()
#         ulabels = list(set(node_info.category))
#         if known_int_emb_dict is not None:
#             known_int_emb_df = pd.DataFrame([(curie_id, array) for curie_id, array in known_int_emb_dict.items()]).rename(columns={0:'id',1:'array'})
#             known_int_emb_df = known_int_emb_df.merge(node_info,on='id').reset_index(drop=True)
#             category_has_known_init_emb = set(known_int_emb_df['category'])
#             for category in category_has_known_init_emb:
#                 try:
#                     assert known_int_emb_df.loc[known_int_emb_df.category.isin([category]),:].shape[0] == node_info.loc[node_info.category.isin([category]),:].shape[0]
#                 except AssertionError:
#                     print(f"Not all curies with cateogry {category} have known intial embedding")    
#                 curie_ids = known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'id']
#                 curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
#                 init_emb = torch.tensor(np.vstack(list(known_int_emb_df.loc[known_int_emb_df.category.isin([category]),'array'])), dtype=torch.float32)
#                 init_embs[category] = (init_emb, curie_ids)
#                 ulabels.remove(category)

#         for label in ulabels:
#             curie_ids = node_info.loc[node_info.category.isin([label]),'id']
#             curie_ids = torch.tensor(list(map(idx_map.get, curie_ids)))
#             init_emb = torch.normal(0, 1, size=(len(curie_ids), dim), dtype=torch.float32)
#             init_embs[label] = (init_emb, curie_ids)

#         return init_embs
    
#     @staticmethod
#     def _rand_rate(n, drug_list, disease_list, idx_map, all_pairs):

#         random.seed(int(time.time()/100))
#         idtoname = {value:key for key, value in idx_map.items()}

#         # get number of drug and disease ids
#         drug_n = len(drug_list)
#         dis_n = len(disease_list)

#         ## create a check list for all tp an tn pairs
#         check_list_temp = {(all_pairs.loc[index,'source'],all_pairs.loc[index,'target']):1 for index in range(all_pairs.shape[0])}
        
#         random_pairs = []
#         ## create 5 times of user's setting number
#         perms = zip(random.choices(list(range(drug_n)),k=5*n), random.choices(list(range(dis_n)),k=5*n))
#         ## select random pairs
#         count = 0
#         for idx in set(perms):
#             rand_pair = (idtoname[drug_list[idx[0]]], idtoname[disease_list[idx[1]]])
#             if rand_pair in check_list_temp:
#                 next
#             else:
#                 check_list_temp[rand_pair] = 1
#                 random_pairs.append(rand_pair)
#                 count += 1
#             if count == n:
#                 break
#         random_pairs = pd.DataFrame(random_pairs).rename(columns={0:'source', 1:'target'})
#         random_pairs['y'] = 0
        
#         print(f'Number of random pairs: {random_pairs.shape[0]}', flush=True)

#         return random_pairs
    
    
#     def process(self):
#         all_nodes = set()
#         all_nodes.update(set(self.raw_edges.source))
#         all_nodes.update(set(self.raw_edges.target))
#         node_info = self.node_info.set_index('id').loc[all_nodes,:].reset_index()
#         idx_map = {j: i for i, j in enumerate(all_nodes)}
#         edges = np.array(list(map(idx_map.get, np.array(self.raw_edges).flatten())), dtype=np.int32).reshape(np.array(self.raw_edges).shape)

#         init_embs = self._generate_init_emb(idx_map, node_info, dim=self.dim, known_int_emb_dict = self.known_int_emb_dict)
#         edge_index = torch.tensor(edges.T, dtype=torch.long)
#         data = Data(feat=init_embs, edge_index=edge_index)
#         id_to_type = {idx_map[node_info['id'][index]]:node_info['category'][index] for index in range(node_info.shape[0])}
#         typeid = {key:index for index, key in enumerate(init_embs)}
#         map_files = [idx_map, id_to_type, typeid]
        
#         # seed random state from time
#         random_state1 = np.random.RandomState(int(time.time()))
#         self.tp_pairs['y'] = 1
#         self.tn_pairs['y'] = 0
#         all_pairs = pd.concat([self.tp_pairs,self.tn_pairs]).reset_index(drop=True)
#         train_index, val_index = train_test_split(np.array(list(all_pairs.index)), train_size=0.9, random_state=random_state1, shuffle=True, stratify=np.array(list(all_pairs['y'])))
        
#         train_pairs = all_pairs.loc[list(train_index),:].reset_index(drop=True)
#         val_pairs = all_pairs.loc[list(val_index),:].reset_index(drop=True)
        
        
#         N = train_pairs.shape[0]//self.batch_size
#         # seed random state from time
#         random_state2 = np.random.RandomState(int(time.time()))
#         cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

#         train_batch = list()
#         os.mkdir(os.path.join(self.processed_dir, 'train_loaders'))
#         for _, index in cv2.split(np.array(list(train_pairs.index)), np.array(train_pairs['y'])):
#             train_batch += [train_pairs.loc[list(index),:].reset_index(drop=True)]
#         print("", flush=True)
#         print(f"generating batches for train set", flush=True)
#         for i in trange(len(train_batch)):

#             batch_data = train_batch[i]
#             data_set = set()
#             data_set.update(set(batch_data.source))
#             data_set.update(set(batch_data.target))
#             data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#             for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                 adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                 loader = (n_id, adjs)
#             filename = 'train_loader' + '_' + str(i) + '.pkl'
#             with open(os.path.join(self.processed_dir, 'train_loaders', filename), 'wb') as output:
#                 pickle.dump(loader, output)
        
        
#         N = val_pairs.shape[0]//self.batch_size
#         # seed random state from time
#         random_state2 = np.random.RandomState(int(time.time()))
#         cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)  

        
#         val_batch = list()
#         os.mkdir(os.path.join(self.processed_dir, 'test_loaders'))
#         for _, index in cv2.split(np.array(list(val_pairs.index)), np.array(val_pairs['y'])):
#             val_batch += [val_pairs.loc[list(index),:].reset_index(drop=True)]
#         print("", flush=True)
#         print(f"generating batches for test set", flush=True)
#         for i in trange(len(test_batch)):

#             batch_data = val_batch[i]
#             data_set = set()
#             data_set.update(set(batch_data.source))
#             data_set.update(set(batch_data.target))
#             data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#             for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                 adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                 loader = (n_id, adjs)
#             filename = 'val_loader' + '_' + str(i) + '.pkl'
#             with open(os.path.join(self.processed_dir, 'val_loaders', filename), 'wb') as output:
#                 pickle.dump(loader, output)


#         N = self.tp_pairs.shape[0]//self.batch_size
#         # seed random state from time
#         random_state2 = np.random.RandomState(int(time.time()))
#         cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

#         tp_batch = list()
#         os.mkdir(os.path.join(self.processed_dir, 'tp_loaders'))
#         for _, index in cv2.split(np.array(list(self.tp_pairs.index)), np.array(tp_pairs['y'])):
#             tp_batch += [self.tp_pairs.loc[list(index),:].reset_index(drop=True)]
#         print("", flush=True)
#         print(f"generating batches for true positive set", flush=True)
#         for i in trange(len(tp_batch)):

#             batch_data = tp_batch[i]
#             data_set = set()
#             data_set.update(set(batch_data.source))
#             data_set.update(set(batch_data.target))
#             data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#             for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                 adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                 loader = (n_id, adjs)
#             filename = 'tp_loader' + '_' + str(i) + '.pkl'
#             with open(os.path.join(self.processed_dir, 'tp_loaders', filename), 'wb') as output:
#                 pickle.dump(loader, output)
                
#         N = self.tn_pairs.shape[0]//self.batch_size
#         # seed random state from time
#         random_state2 = np.random.RandomState(int(time.time()))
#         cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

#         tn_batch = list()
#         os.mkdir(os.path.join(self.processed_dir, 'tn_loaders'))
#         for _, index in cv2.split(np.array(list(self.tn_pairs.index)), np.array(tn_pairs['y'])):
#             tn_batch += [self.tn_pairs.loc[list(index),:].reset_index(drop=True)]
#         print("", flush=True)
#         print(f"generating batches for true negative set", flush=True)
#         for i in trange(len(tn_batch)):

#             batch_data = tn_batch[i]
#             data_set = set()
#             data_set.update(set(batch_data.source))
#             data_set.update(set(batch_data.target))
#             data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#             for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                 adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                 loader = (n_id, adjs)
#             filename = 'tn_loader' + '_' + str(i) + '.pkl'
#             with open(os.path.join(self.processed_dir, 'tn_loaders', filename), 'wb') as output:
#                 pickle.dump(loader, output)

                
#         # generate random pairs of drug and disease
#         print("", flush=True)
#         print(f"generating random pairs of drugs and diseases", flush=True)
#         disease_list = list(set([node_id for node_id, node_type in id_to_type.items() if node_type=='biolink:Disease' or node_type=='biolink:PhenotypicFeature' or node_type=='biolink:DiseaseOrPhenotypicFeature']))
#         drug_list = list(set([node_id for node_id, node_type in id_to_type.items() if node_type=='biolink:Drug' or node_type=='biolink:ChemicalSubstance' or node_type=='biolink:Metabolite']))
#         random_pairs = self._rand_rate(self.N, drug_list, disease_list, idx_map, all_pairs)
        
#         N = random_pairs.shape[0]//self.batch_size
#         # seed random state from time
#         random_state2 = np.random.RandomState(int(time.time()))
#         cv2 = ms.StratifiedKFold(n_splits=N, random_state=random_state2, shuffle=True)

#         rp_batch = list()
#         os.mkdir(os.path.join(self.processed_dir, 'rp_loaders'))
#         for _, index in cv2.split(np.array(list(random_pairs.index)), np.array(random_pairs['y'])):
#             rp_batch += [random_pairs.loc[list(index),:].reset_index(drop=True)]
#         print("", flush=True)
#         print(f"generating batches for random pair set", flush=True)
#         for i in trange(len(rp_batch)):

#             batch_data = rp_batch[i]
#             data_set = set()
#             data_set.update(set(batch_data.source))
#             data_set.update(set(batch_data.target))
#             data_idx=torch.tensor(list(map(idx_map.get, data_set)), dtype=torch.int32)
#             for _, n_id, adjs in NeighborSampler(data.edge_index, node_idx=data_idx, sizes=self.layer_size, batch_size=len(data_idx), shuffle=False, num_workers=self.worker):
#                 adjs = [(adj.edge_index,adj.size) for adj in adjs]
#                 loader = (n_id, adjs)
#             filename = 'rp_loader' + '_' + str(i) + '.pkl'
#             with open(os.path.join(self.processed_dir, 'rp_loaders', filename), 'wb') as output:
#                 pickle.dump(loader, output)
                
#         batch_set = [train_batch, val_batch, tp_batch, tn_batch, rp_batch]
        
#         with open(os.path.join(self.processed_dir, 'batch_set.pkl'), 'wb') as output:
#             pickle.dump(batch_set, output)

#         with open(os.path.join(self.processed_dir, 'map_files.pkl'), 'wb') as output:
#             pickle.dump(map_files, output)
        
#         torch.save(data, os.path.join(self.processed_dir, 'processed_kg.dataset'))
        
#     def get_dataset(self):
#         data = torch.load(os.path.join(self.processed_dir, 'processed_kg.dataset'))
#         return data

#     def get_mapfiles(self):
#         mapfiles = pickle.load(open(os.path.join(self.processed_dir, 'map_files.pkl'), 'rb'))
#         return mapfiles
    
#     def get_batch_set(self):
#         batch_set = pickle.load(open(os.path.join(self.processed_dir, 'batch_set.pkl'), 'rb'))
#         return batch_set

#     def get_train_loader(self):
#         train_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'train_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(train_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
#     def get_val_loader(self):
#         val_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'val_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(val_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
#     def get_tp_loader(self):
#         tp_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'tp_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(tp_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
    
#     def get_tn_loader(self):
#         tn_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'tn_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(tn_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)

#     def get_rp_loader(self):
#         rp_loaders_path = sorted(glob(os.path.join(self.processed_dir, 'rp_loaders', '*.pkl')), key=lambda item: int(item.split('_')[-1].split('.')[0]))
#         return DataLoader(DataWrapper(rp_loaders_path), batch_size=1, shuffle=False, num_workers=self.worker)
