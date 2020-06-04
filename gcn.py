import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, vocab_size, nfeat, nhid, nclass, dropout, features_index, adj):
        super(GCN, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(size=(vocab_size, nfeat)))
        nn.init.normal(self.embedding.data, std=0.1)
        self.features_index = features_index
        self.adj = adj
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x_index):
        features = []
        for index in self.features_index:
            feature = torch.sum(self.embedding[index,:], 0).float().view(1,-1)/len(index)
            features.append(feature)
        x = torch.cat([feature for feature in features], 0)
        x = F.relu(self.gc1(x, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)
        # return F.log_softmax(x, dim=1)
        return x[x_index]
