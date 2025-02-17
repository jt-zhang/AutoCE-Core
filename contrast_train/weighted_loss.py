
import torch
from torch import nn


class WeightedLoss(nn.Module):
    def __init__(self,mag,tau):
        self.mag = mag
        self.tau = tau

    def loss(self,outputs,labels):
        norm2labels = 1. * labels / (torch.norm(labels, 2, 1, keepdim=True).expand_as(labels) + 1e-12)
        sim_mat = torch.matmul(norm2labels, torch.t(norm2labels)) 
        sim_mat = (sim_mat - sim_mat.min() ) / (sim_mat.max()-sim_mat.min() )
        eud = torch.norm(outputs[:, None]-outputs, dim=2, p=2)
        eud = eud/eud.max()
        # print(sim_mat)

        loss = list()
        batch_size = labels.size(0)
        for i in range(batch_size):
            pos_pair_ = sim_mat[i] > self.tau
            neg_pair_ = sim_mat[i] <= self.tau
            distance = eud[i]+sim_mat[i]
            pos_loss = torch.log(torch.sum(torch.exp(distance[pos_pair_]) ) )
            neg_loss = torch.log(torch.sum(torch.exp(self.mag - distance[neg_pair_]) ) )
            if neg_loss<0:
                neg_loss = 0
            if pos_loss<0:
                pos_loss = 0
            loss.append(pos_loss + neg_loss)
            # print(i,float(pos_loss))
            # print(i,float(neg_loss))

        loss = sum(loss) / batch_size
        return loss
