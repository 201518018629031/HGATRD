from gcn import GCN
from gat import SpGAT
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, nfeat, nhid, gat_hidden_dim, joint_dim, features_index, tweet_word_adj, user_tweet_adj, nclass, dropout, alpha):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.nfeat = nfeat
        self.nhid = nhid
        self.gat_hidden_dim = gat_hidden_dim
        self.joint_dim = joint_dim
        self.features_index = features_index
        # self.features = features
        self.tweet_word_adj = tweet_word_adj.cuda()
        self.user_tweet_adj = user_tweet_adj.cuda()
        # self.wV = tweet_word_adj.shape[0]
        self.uV = user_tweet_adj.shape[0]

        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        nn.init.xavier_uniform_(self.user_tweet_embedding.weight)
        self.word_embedding = nn.Parameter(torch.zeros(size=(vocab_size, nfeat)))
        nn.init.normal(self.word_embedding.data, std=0.1)

        self.nclass = nclass
        self.dropout = dropout
        self.alpha = alpha

        # self.gcn = GCN(vocab_size=vocab_size, nfeat=nfeat, nhid=nhid, nclass=joint_dim, dropout=dropout, features_index=features_index, adj=tweet_word_adj)
        # self.twt_gat = SpbaseGAT(vocab_size=vocab_size, nfeat=nfeat, features_index=features_index, hidden=gat_hidden_dim, adj=tweet_word_adj, n_output=joint_dim, dropout=dropout, alpha=alpha)
        # self.tut_gat = SpGAT(nfeat=nfeat, uV=self.uV, hidden=gat_hidden_dim, adj=user_tweet_adj, n_output=joint_dim, dropout=dropout, alpha=alpha)
        self.twt_gat = SpGAT(nfeat=nfeat, hidden=gat_hidden_dim, n_output=joint_dim, dropout=dropout, alpha=alpha)
        self.tut_gat = SpGAT(nfeat=nfeat, hidden=gat_hidden_dim, n_output=joint_dim, dropout=dropout, alpha=alpha)

        # self.lamda = nn.Parameter(torch.rand(1))
        self.weight_W = nn.Parameter(torch.Tensor(joint_dim, joint_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(joint_dim, 1))
        # self.weight_proj = nn.Parameter(torch.Tensor(joint_dim, joint_dim))
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        # self.out1 = nn.Linear(2*joint_dim, joint_dim)
        # self.out1 = nn.Linear(joint_dim, 100)
        # self.out2 = nn.Linear(100, nclass)
        # self.relu = nn.ReLU()
        self.out = nn.Linear(joint_dim, nclass)
        self.init_weight()
        print(self)

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.out.weight)
        # torch.nn.init.xavier_normal_(self.out1.weight)
        # torch.nn.init.xavier_normal_(self.out2.weight)

    def forward(self, tw_graph_idx, ut_graph_idx):

        # tw_vector = self.gcn(tw_graph_idx)
        # ut_vector = self.gat(ut_graph_idx)
        # tw_vector = self.twt_gat(tw_graph_idx)
        # tu_vector = self.tut_gat(ut_graph_idx)
        # features = torch.cat((torch.unsqueeze(tw_vector,dim=0), torch.unsqueeze(tu_vector, dim=0)),dim=0)
        # u = torch.tanh(torch.matmul(features, self.weight_W))
        # att = torch.matmul(u, self.weight_proj)
        # att_score = F.softmax((torch.sum(att, dim=1)/features.shape[1]),dim=0)
        # score_feature = (att_score.repeat(1,features.shape[2]).view(-1,1,features.shape[2]).repeat(1,features.shape[1],1)) * features
        twt_X_list = []
        for index in self.features_index:
            feature = torch.sum(self.word_embedding[index,:], 0).float().view(1,-1)/len(index)
            twt_X_list.append(feature)
        twt_X = torch.cat([feature for feature in twt_X_list], 0)
        tw_X = self.twt_gat(twt_X, self.tweet_word_adj)

        tut_X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        tu_X = self.tut_gat(tut_X, self.user_tweet_adj)

        u_tw = torch.tanh(torch.matmul(tw_X, self.weight_W))
        u_tu = torch.tanh(torch.matmul(tu_X, self.weight_W))
        att_tw = torch.mean(torch.matmul(u_tw, self.weight_proj), dim=0)
        att_tu = torch.mean(torch.matmul(u_tu, self.weight_proj), dim=0)
        att_score = F.softmax(torch.cat((att_tw,att_tu), dim=0), dim=0)
        # att_score = F.softmax(torch.cat((torch.unsqueeze(att_tw,dim=0), torch.unsqueeze(att_tu,dim=0)),dim=0),dim=0)

        out_features = att_score[0] * tw_X[tw_graph_idx,:] + att_score[1] * tu_X[ut_graph_idx,:]
        # features = torch.cat((tw_vector, ut_vector),dim=1)
        # features = self.lamda*tw_vector + (1 - self.lamda)*ut_vector
        # f1 = F.dropout(self.relu(self.out1(out_features)), self.dropout, training=self.training)
        # output = self.out2(f1)
        output = self.out(out_features)
        return F.log_softmax(output, dim=1)
