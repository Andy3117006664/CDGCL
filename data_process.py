from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import pickle
import scipy.sparse as sp
from pprint import pprint

import math
import numpy as np
import torch_geometric
#import dgl

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Actor, Reddit
from torch_geometric.transforms import to_undirected
from utils import *
import networkx as nx
from torch_geometric.transforms import NormalizeFeatures
from model.model_cond import *
from torch_geometric.data import Data



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    if sp.issparse(features):
        return features.todense()
    else:
        return features


class PyG(Data):
    def __init__(self, x=None, edge_index=None, y=None, **kwargs):
        super(PyG, self).__init__(x=x, edge_index=edge_index, y=y, **kwargs)
        self.adj = None
        self.task = None
        self.path = None

    def initialize(self, args, path, ds):
        super(PyG, self).__init__()
        self.path = path
        self.task = args.task

        #意思还能实现node classification？
        if self.task == 'node':
            transform = NormalizeFeatures()
        else:
            transform = None

        if ds in ['Cora', 'Citeseer']:
            data = Planetoid(root=self.path, name=ds, transform=transform)[0]
            self.train = data['train_mask']
            self.val = data['val_mask']
            self.test = data['test_mask']

            self.train = torch.where(self.train==1.0)[0]
            self.val = torch.where(self.val==1.0)[0]
            self.test = torch.where(self.test==1.0)[0]
            self.un_lbl = torch.where(self.train==0.0)[0]

           
        elif ds in ['CS', 'Physics']:
            data = Coauthor(root=self.path, name=ds, transform=transform)[0]

            num_ent = data['y'].shape[0]
            np.random.seed(0)
            idx = np.arange(num_ent)
            np.random.shuffle(idx)

            self.train = idx[:int(0.2*num_ent)]
            self.val = idx[int(0.2*num_ent):int(0.4*num_ent)]
            self.test = idx[int(0.4*num_ent):]

            self.un_lbl = np.setdiff1d(idx, self.train)

            self.train = torch.LongTensor(self.train)
            self.val = torch.LongTensor(self.val)
            self.test = torch.LongTensor(self.test)

        elif ds in ['Computers']:
            data = Amazon(root=self.path, name=ds, transform=transform)[0]

        elif ds in ['Actor']:
            data = Actor(root=os.path.join(self.path, 'Actor'), transform=transform)[0]

        print(data)
        
        
        self.feat = data.x
        edges = data.edge_index
        self.edges = edges
        self.num_ent = self.feat.shape[0]

        # # 初始化扩散模型
        # self.diffusion = Diffusion_Cond(
        #     in_feat=self.feat.shape[1],
        #     out_feat=args.nhid,
        #     args=args,
        #     device= self.feat.device,
        #     y=None
        # )

        if self.task == 'node':
           
            self.adj = sp.coo_matrix(
                            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(self.num_ent, self.num_ent),
                            dtype=np.float32)

            # to undirected
            #self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
            self.adj = self.adj + sp.eye(self.num_ent)
            self.adj = self.adj.todense()
            self.adj = normalize_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj)

            self.y = data['y']
            
            f_label = int(max(self.y)+1)
            self.f_label = torch.full(self.y.shape, f_label)

        elif self.task == 'link':
            
            data_list = '_triples_{:d}.txt'.format(neg_num_samp)

            if not os.path.exists(os.path.join(self.path, ds, 'train'+data_list)):
                np.random.seed(0)
                idx = np.arange(edges.shape[0])
                np.random.shuffle(idx)

                num_train = idx[:int(0.9*idx.shape[0])]
                num_val = idx[int(0.9*idx.shape[0]):int(0.95*idx.shape[0])]
                num_test = idx[int(0.95*idx.shape[0]):]

                self.links = dict()
                
                train = edges[num_train].tolist()
                #print(train[0])

                val = edges[num_val]
                test = edges[num_test]
                
                val_test = np.concatenate((val, test))
                inv_val_test = val_test[:, [1,0]].tolist()

                for e in inv_val_test:
                    if e in train:
                        train.remove(e)
                #print(len(train))

                self.links['train'] = np.asarray(train)
                self.links['val'] = val
                self.links['test'] = test

                create_train_test_split(self.links, self.path, num=self.num_ent, name=args.dataset, data_list=data_list)

            self.t = ddict(list)
            for split in ['train', 'test', 'val']:
                self.t[split] = np.genfromtxt(os.path.join(self.path, ds, split+data_list), \
                                                    delimiter=' ', dtype=int)
                self.t[split] = self.t[split][:,:11]

            self.adj = sp.coo_matrix(
                            (np.ones(self.t['train'].shape[0]), (self.t['train'][:, 0], self.t['train'][:, 1])),
                            shape=(self.num_ent, self.num_ent),
                            dtype=np.float32)

            # to undirected
            self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
            self.adj = self.adj + sp.eye(self.num_ent)
            self.adj = self.adj.todense()
            adj1 = self.adj
            self.adj = normalize_adj(self.adj)

            self.adj = torch.FloatTensor(self.adj)

    # def augment_features(self, labels=None):
    #     """特征增强方法"""
    #     # print("label:",self.y.shape)
    #     # print("feat:",self.feat.shape)
    #     return self.diffusion.augment_features(self.feat, labels)
        
    # def augment_edges(self, edge_index):
    #     """边增强方法"""
    #     return self.diffusion.augment_edges(edge_index)



            
    def to_cuda(self):
        self.feat = self.feat.cuda()
        self.adj = self.adj.cuda()

        if self.task == 'node':
            self.label = self.label.cuda()
            self.f_label = self.f_label.cuda()

    def to_device(self, device):
        self.feat = self.feat.to(device)
        self.adj = self.adj.to(device)
       
DATASETS = {
    'Cora': PyG, 
    'Citeseer': PyG,
    'CS': PyG,
    'Actor': PyG
    }



def get_dataset(args, name, path='data'):
    if name not in DATASETS:
        raise ValueError("Dataset is not supported")
        
    # 创建空的 PyG 实例
    pyg_instance = PyG()
    
    # 使用 initialize 方法加载数据
    pyg_instance.initialize(args, path, name)
    
    return pyg_instance



if __name__ == "__main__":
    
    args = parse_args()
    args.max_l += 1
    print(args)
    
    np.random.seed(args.seed)
    seed = np.random.choice(100, args.n_runs, replace=False)
    print('Seed: ', seed)

    print('Processing data ...')
    data = get_dataset(args, args.dataset)