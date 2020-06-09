import numpy as np
import scipy.sparse as sp
import torch
import joblib as jlb
import logging, sys


#logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_vocab_len(dataset="twitter15"):
    with open('dataset/{}/{}_vocab.txt'.format(dataset, dataset), 'rb') as f:
        lines = f.readlines()
        vocab_size = len(lines)

    return vocab_size

def load_data(dataset="twitter15"):
    """
    Load citation network dataset (twitter15, twitter16, and weibo)
    Loads input corpus from gcn/data directory
    ind.dataset.features => the feature vectors of the nodes (tweets and words) as scipy.sparse.csr.csr_matrix object;
    ind.dataset.train => the indices of training tweets in nodes;
    ind.dataset.dev => the indices of dev tweets in nodes;
    ind.dataset.test => the indices of test tweets in nodes;
    ind.dataset.labels => the one-hot labels of the all nodes as numpy.ndarray object;
    ind.dataset.adj => adjacency matrix of words/tweets nodes as scipy.sparse.csr.csr_matrix object;
    All objects above must be saved using python pickle module.
    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    print('Loading {} dataset...'.format(dataset))

    names = ['features_index', 'train', 'dev', 'test', 'labels', 'adj']
    objects = []
    for i in range(len(names)):
        with open('dataset/{}/ind.{}.{}'.format(dataset, dataset, names[i]), 'rb') as f:
            objects.append(jlb.load(f))
            # if sys.version_info > (3, 0):
            #     objects.append(jlb.load(f, encoding='latin1'))
            # else:
            #     objects.append(jlb.load(f))

    features_index, train_ids, dev_ids, test_ids, labels, adj = tuple(objects)

    # logger.info('features.shape:{}, the length of train_ids:{}, the length of dev_ids:{}, the length of test_ids:{}, labels.shape:{}'.format(features.shape,len(train_ids),len(dev_ids),len(test_ids),labels.shape))
    logger.info('features_index.shape:{}, the length of train_ids:{}, the length of dev_ids:{}, the length of test_ids:{}, labels.shape:{}'.format(len(features_index),len(train_ids),len(dev_ids),len(test_ids),labels.shape))


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features.tolil())
    # features = features.tolil()
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(train_ids)
    idx_val = torch.LongTensor(dev_ids)
    idx_test = torch.LongTensor(test_ids)

    return adj, features_index, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_user_tweet_graph(dataset, elapsed_time, tweets_count):
    print('Loading the user_tweet graph of {} dataset ...'.format(dataset))

    # if elapsed_time == 3000 and tweets_count == 500:
    #     path_content = 'dataset/{}/{}_graph.txt'.format(dataset,dataset)
    # elif elapsed_time != 3000 and tweets_count == 500:
    #     path_content = 'dataset/{}/{}_graph_et{}.txt'.format(dataset, dataset, elapsed_time)
    # elif elapsed_time == 3000 and tweets_count != 500:
    #     path_content = 'dataset/{}/{}_graph_tc{}.txt'.format(dataset, dataset, tweets_count)
    #
    # X_tids = []
    # X_uids = []
    # with open('dataset/{}/{}.idx.txt'.format(dataset, dataset), 'r') as f:
    #     line = f.readline()
    #     X_tids = line.split()
    #
    # with open(path_content, 'r', encoding='utf-8') as input:
    #     relation = []
    #     for line in input.readlines():
    #         tmp = line.strip().split()
    #         src = tmp[0]
    #         X_uids.append(src)
    #
    #         for dst_ids_ws in tmp[1:]:
    #             dst, w = dst_ids_ws.split(":")
    #             X_uids.append(dst)
    #             relation.append([src, dst, w])
    #
    # X_id = list(set(X_tids + X_uids))
    # num_node = len(X_id)
    # print(num_node)
    # X_id_dic = {id:i for i, id in enumerate(X_id)}
    #
    # relation = np.array([[X_id_dic[tup[0]], X_id_dic[tup[1]], tup[2]] for tup in relation])
    # relation = build_symmetric_adjacency_matrix(relation, shape=(num_node, num_node))
    # train_idx = []
    # dev_idx = []
    # test_idx = []
    # with open('dataset/{}/{}.train'.format(dataset, dataset), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         idx = line.strip().split()[0]
    #         train_idx.append(X_id_dic[idx])
    # with open('dataset/{}/{}.dev'.format(dataset, dataset), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         idx = line.strip().split()[0]
    #         dev_idx.append(X_id_dic[idx])
    # with open('dataset/{}/{}.test'.format(dataset, dataset), 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         idx = line.strip().split()[0]
    #         test_idx.append(X_id_dic[idx])

    names = ['train', 'dev', 'test', 'adj']
    objects = []
    for i in range(len(names)):
        if elapsed_time == 3000 and tweets_count == 500:
            path = 'dataset/{}/ind.{}.user.tweet.{}'.format(dataset, dataset, names[i])
        elif elapsed_time != 3000 and tweets_count == 500:
            path = 'dataset/{}/ind.{}.user.tweet.{}.et{}'.format(dataset, dataset, names[i], elapsed_time)
        elif elapsed_time == 3000 and tweets_count != 500:
            path = 'dataset/{}/ind.{}.user.tweet.{}.tc{}'.format(dataset, dataset, names[i], tweets_count)
        with open(path, 'rb') as f:
            objects.append(jlb.load(f))
    train_idx, dev_idx, test_idx, adj = tuple(objects)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    relation = sparse_mx_to_torch_sparse_tensor(adj)

    train_idx = torch.LongTensor(train_idx)
    dev_idx = torch.LongTensor(dev_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, dev_idx, test_idx, relation

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def build_symmetric_adjacency_matrix(edges, shape):

    adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape, dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj.tocoo()

def evaluation_4class(prediction, y): # 4 dim
    prediction = prediction.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0
    e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(y)):
        y_i, p_i = list(y[i]), list(prediction[i])
        ##RMSE
        for j in range(len(y_i)):
            RMSE += (y_i[j]-p_i[j])**2
        RMSE1 += (y_i[0]-p_i[0])**2
        RMSE2 += (y_i[1]-p_i[1])**2
        RMSE3 += (y_i[2]-p_i[2])**2
        RMSE4 += (y_i[3]-p_i[3])**2
        ## Pre, Recall, F
        Act = str(y_i.index(max(y_i))+1)
        Pre = str(p_i.index(max(p_i))+1)

        #print y_i, p_i
        #print Act, Pre
        ## for class 1
        if Act == '1' and Pre == '1': TP1 += 1
        if Act == '1' and Pre != '1': FN1 += 1
        if Act != '1' and Pre == '1': FP1 += 1
        if Act != '1' and Pre != '1': TN1 += 1
        ## for class 2
        if Act == '2' and Pre == '2': TP2 += 1
        if Act == '2' and Pre != '2': FN2 += 1
        if Act != '2' and Pre == '2': FP2 += 1
        if Act != '2' and Pre != '2': TN2 += 1
        ## for class 3
        if Act == '3' and Pre == '3': TP3 += 1
        if Act == '3' and Pre != '3': FN3 += 1
        if Act != '3' and Pre == '3': FP3 += 1
        if Act != '3' and Pre != '3': TN3 += 1
        ## for class 4
        if Act == '4' and Pre == '4': TP4 += 1
        if Act == '4' and Pre != '4': FN4 += 1
        if Act != '4' and Pre == '4': FP4 += 1
        if Act != '4' and Pre != '4': TN4 += 1
    ## print result
    Acc_all = round( float(TP1+TP2+TP3+TP4)/float(len(y)+e), 4 )
    Acc1 = round( float(TP1+TN1)/float(TP1+TN1+FN1+FP1+e), 4 )
    Prec1 = round( float(TP1)/float(TP1+FP1+e), 4 )
    Recll1 = round( float(TP1)/float(TP1+FN1+e), 4 )
    F1 = round( 2*Prec1*Recll1/(Prec1+Recll1+e), 4 )

    Acc2 = round( float(TP2+TN2)/float(TP2+TN2+FN2+FP2+e), 4 )
    Prec2 = round( float(TP2)/float(TP2+FP2+e), 4 )
    Recll2 = round( float(TP2)/float(TP2+FN2+e), 4 )
    F2 = round( 2*Prec2*Recll2/(Prec2+Recll2+e), 4 )

    Acc3 = round( float(TP3+TN3)/float(TP3+TN3+FN3+FP3+e), 4 )
    Prec3 = round( float(TP3)/float(TP3+FP3+e), 4 )
    Recll3 = round( float(TP3)/float(TP3+FN3+e), 4 )
    F3 = round( 2*Prec3*Recll3/(Prec3+Recll3+e), 4 )

    Acc4 = round( float(TP4+TN4)/float(TP4+TN4+FN4+FP4+e), 4 )
    Prec4 = round( float(TP4)/float(TP4+FP4+e), 4 )
    Recll4 = round( float(TP4)/float(TP4+FN4+e), 4 )
    F4 = round( 2*Prec4*Recll4/(Prec4+Recll4+e), 4 )

    microF = round( (F1+F2+F3+F4)/4,5 )
    RMSE_all = round( ( RMSE/len(y) )**0.5, 4)
    RMSE_all_1 = round( ( RMSE1/len(y) )**0.5, 4)
    RMSE_all_2 = round( ( RMSE2/len(y) )**0.5, 4)
    RMSE_all_3 = round( ( RMSE3/len(y) )**0.5, 4)
    RMSE_all_4 = round( ( RMSE4/len(y) )**0.5, 4)
    RMSE_all_avg = round( ( RMSE_all_1+RMSE_all_2+RMSE_all_3+RMSE_all_4 )/4, 4)
    # return ['acc:', Acc_all, 'Favg:',microF, RMSE_all, RMSE_all_avg,
    #         'C1:',Acc1, Prec1, Recll1, F1,
    #         'C2:',Acc2, Prec2, Recll2, F2,
    #         'C3:',Acc3, Prec3, Recll3, F3,
    #         'C4:',Acc4, Prec4, Recll4, F4]

    return Acc_all, microF, RMSE_all, RMSE_all_avg, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3,Recll3, F3, Acc4, Prec4, Recll4, F4

def convert_to_one_hot(y, C):
    # return np.eye(C)[y.reshape(-1)]
    return torch.zeros(y.shape[0], C).scatter_(1, y, 1)
