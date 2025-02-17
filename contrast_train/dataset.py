# dataset.py
import math
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
from dgl.dataloading import GraphDataLoader
import scipy.sparse as sp
from utils import preprocess
from dgl.data import DGLDataset
import dgl.function as fn
import copy


class MyDataset(DGLDataset):
    # feat_dict: dict(dict(array))  A_dict: dict(array)  label_acc_dict: dict(list[bayescard,deepdb,mscn,naru,nn,xgb])
    def __init__(self, feat_dict,A_dict,W_dict,label_acc_dict,label_eff_dict,acc_wgt=1.0):
        self.feat_dict = feat_dict
        self.A_dict = A_dict
        self.W_dict = W_dict
        self.label_acc_dict = label_acc_dict
        self.label_eff_dict = label_eff_dict
        self.acc_wgt = acc_wgt
        self.gclasses = len(label_acc_dict[0])
        self.dim_nfeats = len(feat_dict[0][0])
        super(MyDataset, self).__init__(name='card')
        # print('done init')
        # self.graphs = graphs
        # self.label = label

    def getgrh_lab(self, feat_dict ,A_dict, label_acc_dict, label_eff_dict, acc_wgt=1.0):
        len_dataset = 1200 # len(A_dict)
        graphs = []
        labels = []

        for i in range(len_dataset):
            A = A_dict[i]
            spmat = sp.csr_matrix(A)
            g = dgl.from_scipy(spmat) 
            g = dgl.add_self_loop(g)  # 
            feat = np.zeros((len(feat_dict[i]),len(feat_dict[i][0])))
            for g_ni in range(len(feat_dict[i])):
                feat[g_ni] = feat_dict[i][g_ni]
            feat = torch.from_numpy(feat)
            g.ndata['attr'] = feat
            label = acc_wgt*np.array(label_acc_dict[i]) + (1-acc_wgt)*np.array(label_eff_dict[i])
            label = torch.from_numpy(label)

            graphs.append(g)
            labels.append(label)

        return graphs, labels

    def edges_wgt(self):
        for i in range(1200):  # (len(self.W_dict)):
            edges_feat = torch.tensor([self.W_dict[i][row][col] for row in range(self.W_dict[i].shape[0]) for col in range(self.W_dict[i].shape[1]) if self.W_dict[i][row][col] != 0])
            # print('edges_feat:',i,edges_feat)
            # print(f'self.graphs[{i}].edges():',self.graphs[i].edges())
            edges_feat= torch.tensor(edges_feat, dtype=torch.float32)  # modify
            # gft_copy = copy.deepcopy(self.graphs[i].ndata['attr'])
            # print('edges_feat:',edges_feat)
            edges_feat = torch.cat((edges_feat,torch.ones(self.W_dict[i].shape[0])))  
            self.graphs[i].edata['join_sel'] = edges_feat
            # self.graphs[i].update_all(fn.u_mul_e('attr', 'join_sel', 'm'), fn.sum('m', 'attr'))  # sum？ mean max min
            # self.graphs[i].ndata['attr'] += 10*gft_copy 
            # self.graphs[i].ndata['attr'] /= 10  # 

        return self.graphs

    def process(self):
        # mat_path = self.raw_path + '.mat'
        # self.graphs, self.label = self._load_graph(mat_path)
        self.graphs, self.label = self.getgrh_lab(self.feat_dict, self.A_dict, self.label_acc_dict, self.label_eff_dict, self.acc_wgt)
        self.graphs = self.edges_wgt()  # 

    def __getitem__(self, idx):

        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)
