import sys
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
from dataloader import GINDataLoader
from ginparser import Parser
from gin import GIN
from utils import preprocess,lable2cate,s_max_min,y2Y
from dataset import MyDataset


def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))


def eval_net(args, net, trainloader, validloader):
    net.eval()
    mark = 0
    for data in trainloader:
        
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        feat= torch.tensor(feat, dtype=torch.float32)  # modify
        outputs = net(graphs, feat)
        if mark == 0:
            candidate = outputs
        else:
            candidate = torch.cat((candidate, outputs), 0)
        if mark == 0:
            c_labels = labels
        else:
            c_labels = torch.cat((c_labels, labels), 0)
        mark += 1

    mark = 0
    for data in validloader:
        graphs, labels = data
        graphs = graphs.to(args.device)
        labels = labels.to(args.device)
        feat = graphs.ndata.pop('attr')
        feat= torch.tensor(feat, dtype=torch.float32)  # modify
        outputs = net(graphs, feat)
        if mark == 0:
            test = outputs
        else:
            test = torch.cat((test, outputs), 0)
        mark += 1

    # print('shape:',test.shape,candidate.shape)
    distance = EuclideanDistances(test, candidate)  # 300*900 
    print('distance:',distance)
    return distance.argmax(dim=1)


def main(args, dataset):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    
    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    # dataset = GINDataset(args.dataset, not args.learn_eps)
    # print('dataset:',dataset)

    trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=False,
        split_name='rand', split_ratio=0.75).train_valid_loader()
    # or split_name='fold10', fold_idx=args.fold_idx  

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, args.hidden_dim,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    model.load_state_dict(torch.load('./model/embedding.pth'))

    res = eval_net(args, model, trainloader, validloader)
    print('res:',res)




if __name__ == '__main__':
    A_dict,W_dict,feat_dict,label_acc_dict,label_eff_dict = preprocess()
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    dataset = MyDataset(feat_dict,A_dict,W_dict,label_acc_dict,label_eff_dict,acc_wgt=args.acc_wgt)
    list_all_label = dataset.label  # list
    main(args, dataset)
