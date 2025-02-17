import torch
from ginparser import Parser
from gin import GIN
args = Parser(description='GIN').args
from dataloader import GINDataLoader
from dataset import MyDataset
from utils import preprocess
import numpy as np
import torch.nn.functional as F
import pickle

A_dict,W_dict,feat_dict,label_acc_dict,label_eff_dict = preprocess()
dataset = MyDataset(feat_dict,A_dict,W_dict,label_acc_dict,label_eff_dict,acc_wgt=args.acc_wgt)

is_cuda = not args.disable_cuda and torch.cuda.is_available()
if is_cuda:
    args.device = torch.device("cuda:" + str(args.device))
    torch.cuda.manual_seed_all(seed=args.seed)
else:
    args.device = torch.device("cpu")
trainloader, validloader = GINDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name='rand', split_ratio=0.834).train_valid_loader()

model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, args.hidden_dim,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)


model.load_state_dict(torch.load('./model/embedding.pth'))
model.eval()

train_vecs = list()
test_vecs = list()

for (graphs, labels) in trainloader:
    # batch graphs will be shipped to device in forward part of model
    labels = labels.to(args.device)
    graphs = graphs.to(args.device)
    feat = graphs.ndata.pop('attr')
    feat = torch.tensor(feat, dtype=torch.float32)  # modify
    outputs = model(graphs, feat)
    for i in range(len(outputs) ):
        train_vecs.append(outputs[i].cpu().detach().numpy())


for (graphs, labels) in validloader:
    # batch graphs will be shipped to device in forward part of model
    labels = labels.to(args.device)
    graphs = graphs.to(args.device)
    feat = graphs.ndata.pop('attr')
    feat = torch.tensor(feat, dtype=torch.float32)  # modify
    outputs = model(graphs, feat)
    for i in range(len(outputs) ):
        test_vecs.append(outputs[i].cpu().detach().numpy())

train_vecs = torch.tensor(train_vecs).to(args.device)
test_vecs = torch.tensor(test_vecs).to(args.device)

# print(train_vecs, test_vecs)

# for i in range(len(test_vecs)):
res = []
for i in range(len(test_vecs)):
    eud = F.pairwise_distance(test_vecs[i],train_vecs,p=2)
    res.append(int(eud.argmin()))

f_res = open('../exp/res/res.list', 'wb')
pickle.dump(res, f_res)
print(res)
f_res.close()

