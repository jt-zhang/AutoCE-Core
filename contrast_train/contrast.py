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
from weighted_loss import WeightedLoss


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        graphs = graphs.to(args.device)
        feat = graphs.ndata.pop('attr')
        feat = torch.tensor(feat, dtype=torch.float32)  # modify
        
        outputs = net(graphs, feat)
        
        labels = torch.where(torch.isnan(labels), torch.full_like(labels, 0.5), labels)

        bs = len(outputs)

        # o1,o2 = outputs[0:int(bs/2),:], outputs[int(bs/2):int(bs),:]
        # label1,label2 = labels[0:int(bs/2),:], labels[int(bs/2):int(bs),:]

        # loss = criterion(outputs.float(), labels.float())
        Loss = WeightedLoss(args.LossM, args.tau)
        loss = Loss.loss(outputs,labels)
        
        # running_loss += loss
        running_loss += loss.item()
        # print('runing_loss:', running_loss)

        # backprop
        optimizer.zero_grad()
        
        loss.backward()
        # outputs.sum().backward()
        # for name, parms in net.named_parameters(): # View network parameters and gradient information
            # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', parms.grad) # torch.mean(parms.grad)
        optimizer.step()
        
        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


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
        seed=args.seed, shuffle=True,
        split_name='rand', split_ratio=0.83).train_valid_loader()
    # or split_name='fold10', fold_idx=args.fold_idx  

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, args.hidden_dim,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    criterion = torch.nn.MSELoss()  #  nn.CrossEntropyLoss()  # default reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    # tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    # vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    # lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch in tqdm(range(args.epochs)):# zip(tbar, vbar, lrbar):

        run_loss = train(args, model, trainloader, optimizer, criterion, epoch)
        print('loss:', run_loss)
        scheduler.step()

    torch.save(model.state_dict(), './model/embedding.pth')


if __name__ == '__main__':
    A_dict,W_dict,feat_dict,label_acc_dict,label_eff_dict = preprocess()
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)
    dataset = MyDataset(feat_dict,A_dict,W_dict,label_acc_dict,label_eff_dict,acc_wgt=args.acc_wgt)

    main(args, dataset)
