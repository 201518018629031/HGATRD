import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from layers import SpGraphAttentionLayer

class SpGAT(nn.Module):
    def __init__(self, nfeat, hidden=16, nb_heads=8, n_output=100, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # self.uV = uV
        # self.adj = adj.cuda()
        # self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        # init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features = nfeat,
                                                        out_features= hidden,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nb_heads)])

        self.out_att = SpGraphAttentionLayer(hidden * nb_heads,
                                              n_output,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, X, adj):
        # X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        X = self.dropout(X)
        X = torch.cat([att(X, adj) for att in self.attentions], dim=1)
        X = self.dropout(X)

        X = F.elu(self.out_att(X, adj))
        # X_ = X[X_tid]
        return X

# class SpbaseGAT(nn.Module):
#     def __init__(self, vocab_size, nfeat, features_index, adj, hidden=16, nb_heads=8, n_output=100, dropout=0.5, alpha=0.3):
#         """Sparse version of GAT."""
#         super(SpbaseGAT, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.features_index = features_index
#         self.embedding = nn.Parameter(torch.zeros(size=(vocab_size, nfeat)))
#         nn.init.normal(self.embedding.data, std=0.1)
#         # self.X = features.cuda()
#         self.adj = adj.cuda()
#
#         self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features = nfeat,
#                                                         out_features= hidden,
#                                                         dropout=dropout,
#                                                         alpha=alpha,
#                                                         concat=True) for _ in range(nb_heads)])
#
#         self.out_att = SpGraphAttentionLayer(hidden * nb_heads,
#                                               n_output,
#                                              dropout=dropout,
#                                              alpha=alpha,
#                                              concat=False)
#
#     def forward(self):
#         features = []
#         for index in self.features_index:
#             feature = torch.sum(self.embedding[index,:], 0).float().view(1,-1)/len(index)
#             features.append(feature)
#         X = torch.cat([feature for feature in features], 0)
#         X = self.dropout(X)
#
#         X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
#         X = self.dropout(X)
#
#         X = F.elu(self.out_att(X, self.adj))
#
#         return X
