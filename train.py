from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random, os

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, load_vocab_len, load_user_tweet_graph, accuracy, evaluation_4class, convert_to_one_hot
from models import Model
from sklearn.metrics import accuracy_score, classification_report

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--dataset', type=str, default='twitter15',
                    help='the dataset name: twitter15, twitter16, and weibo (default: twitter15)')
parser.add_argument('--embed_size', type=int, default=300,
                    help='the dimension of word embedding (default: 300)')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--gat_hidden_dim', type=int, default=16,
                    help='Number of gat hidden units. ')
parser.add_argument('--joint_dim', type=int, default=300,
                    help='dimension of each module output')
parser.add_argument('--elapsed_time',type=int, default=3000,
                    help='the elapsed time after source tweet posted (0, 60(1h), 120(2h), 240(4h), 480(8h), 720(12h), 1440(24h), 2160(36h), default: 3000 represents all)')
parser.add_argument('--tweets_count', type=int, default=500,
                    help='the tweets count after source tweet posted (0, 10, 20, 40, 60, 80, 200, 300, default: 500 represents all)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
# parser.add_argument('--target_names', nargs='+', default=['NR','FR','TR','UR'],
#                     help='the label of rumors (twitter:NR,FR,TR,UR, weibo:FR,TR; default:twitter)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate of GAT(1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.3,
                    help='Alpha for the leaky_relu of GAT.')
parser.add_argument('--filename', type = str, default = "",
                                    help='output file')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
target_names = ['NR','FR','TR','UR']
if args.dataset == 'weibo':
    target_names = ['NR','FR']

# def seed_everything(seed=2040):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

random.seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
# tweet_word_adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
tweet_word_adj, features_index, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
vocab_size = load_vocab_len(args.dataset) + 1
train_idx, dev_idx, test_idx, user_tweet_adj = load_user_tweet_graph(args.dataset, args.elapsed_time, args.tweets_count)

# if args.cuda:
#     tweet_word_adj = tweet_word_adj.cuda()
# Model and optimizer
# model = GCN(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)
model = Model(vocab_size=vocab_size,
            nfeat=args.embed_size,
            nhid=args.hidden,
            gat_hidden_dim=args.gat_hidden_dim,
            joint_dim=args.joint_dim,
            features_index=features_index,
            tweet_word_adj=tweet_word_adj,
            user_tweet_adj=user_tweet_adj,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    train_idx = train_idx.cuda()
    dev_idx = dev_idx.cuda()
    test_idx = test_idx.cuda()

def train(epoch, best_acc, patience):
    # t = time.time()
    model.train()
    total_iters = len(idx_train)//args.batch_size + 1
    loss_accum = 0
    avg_acc = 0
    idx_list = np.arange(len(idx_train))
    for i in range(total_iters):
        # selected_idx = np.random.permutation(len(idx_train))[:args.batch_size]
        selected_idx = idx_list[(i*args.batch_size):((i+1)*args.batch_size)]
        if len(selected_idx) == 0:
            continue
        batch_idx_train = torch.LongTensor([idx_train[id] for id in selected_idx])
        batch_train_idx = torch.LongTensor([train_idx[id] for id in selected_idx])
        output = model(batch_idx_train, batch_train_idx)
        loss_train = F.nll_loss(output, labels[batch_idx_train])
        batch_labels = labels[batch_idx_train]

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        corrects = (torch.max(output, 1)[1].view(len(batch_idx_train)).data == batch_labels.data).sum()
        accuracy = 100*corrects/len(batch_idx_train)

        if i > 0 and i % 100 == 0:
            best_acc, patience = evaluate(best_acc, patience)
            model.train()
        avg_acc += accuracy
        print('Batch [{}] - loss:{:.6f} acc:{:.4f}% ({}/{})'.format(i, loss_train.item(), accuracy, corrects, len(batch_idx_train)))
        loss = loss_train.detach().cpu().numpy()
        loss_accum += loss

        #report
        # pbar.set_description('epoch {}'.format(epoch))

    average_loss = loss_accum/total_iters
    average_acc = avg_acc/total_iters
    print("loss training: {:.6f} average_acc: {:.6f}".format(average_loss, average_acc))

    return best_acc, patience


def pass_data_iteratively(tw_idx_list, tu_idx_list, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(tw_idx_list))
    for i in range(0, len(tw_idx_list), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model(tw_idx_list[sampled_idx], tu_idx_list[sampled_idx]).detach())
    return torch.cat(output, 0)

def adjust_learning_rate(optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr

def evaluate(best_acc, patience):
    # model.eval()
    output = pass_data_iteratively(idx_val, dev_idx)
    predicted = torch.max(output, dim=1)[1]
    y_pred = predicted.data.cpu().numpy().tolist()
    val_labels = labels[idx_val].data.cpu().numpy().tolist()
    acc = accuracy_score(val_labels, y_pred)

    if acc > best_acc:
        best_acc = acc
        patience = 0
        if args.elapsed_time == 3000 and args.tweets_count == 500:
            torch.save(model.state_dict(), 'weights.best.{}'.format(args.dataset))
        elif args.elapsed_time < 3000 and args.tweets_count == 500:
            torch.save(model.state_dict(), 'weights.best.{}.et{}'.format(args.dataset, args.elapsed_time))
        elif args.elapsed_time == 3000 and args.tweets_count < 500:
            torch.save(model.state_dict(), 'weights.best.{}.tc{}'.format(args.dataset, args.tweets_count))
        print(classification_report(val_labels, y_pred, target_names=target_names, digits=5))
        print('Val set acc: {}'.format(acc))
        print('Best val set acc: {}'.format(best_acc))
        print('save model!!!!')
    else:
        patience += 1

    return best_acc, patience


def test():
    # model.eval()
    output = pass_data_iteratively(idx_test, test_idx)
    predicted = torch.max(output, dim=1)[1]
    y_pred = predicted.data.cpu().numpy().tolist()
    test_labels = labels[idx_test].data.cpu().numpy().tolist()
    print('=====================================')
    print(classification_report(test_labels, y_pred, target_names=target_names, digits=5))
    t_labels = convert_to_one_hot(labels[idx_test].unsqueeze(1).cpu(), 4).cuda()
    if args.dataset == 'weibo':
        t_labels = convert_to_one_hot(labels[idx_test].unsqueeze(1).cpu(), 2).cuda()
    result_test = evaluation_4class(output, t_labels)
    return result_test


# Train model
best_acc = 0.0
patience = 0
t_total = time.time()
for epoch in range(1, args.epochs+1):
    print("Epoch {}/{}".format(epoch, args.epochs))
    best_acc, patience = train(epoch, best_acc, patience)
    if epoch >= 10 and patience > 3:
        print('Reload the best model ...')
        if args.elapsed_time == 3000 and args.tweets_count == 500:
            model.load_state_dict(torch.load('weights.best.{}'.format(args.dataset)))
        elif args.elapsed_time < 3000 and args.tweets_count == 500:
            model.load_state_dict(torch.load('weights.best.{}.et{}'.format(args.dataset, args.elapsed_time)))
        elif args.elapsed_time == 3000 and args.tweets_count < 500:
            model.load_state_dict(torch.load('weights.best.{}.tc{}'.format(args.dataset, args.tweets_count)))
        now_lr = adjust_learning_rate(optimizer)
        print(now_lr)
        patience = 0
    best_acc, patience = evaluate(best_acc, patience)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print('Loading model to test set ...')
if args.elapsed_time == 3000 and args.tweets_count == 500:
    model.load_state_dict(torch.load('weights.best.{}'.format(args.dataset)))
elif args.elapsed_time < 3000 and args.tweets_count == 500:
    model.load_state_dict(torch.load('weights.best.{}.et{}'.format(args.dataset, args.elapsed_time)))
elif args.elapsed_time == 3000 and args.tweets_count < 500:
    model.load_state_dict(torch.load('weights.best.{}.tc{}'.format(args.dataset, args.tweets_count)))
result_test = test()
if not args.filename == "":
    with open(args.filename, 'w') as f:
        f.write('the result of test:')
        f.write('acc:{:.4f} Favg:{:.4f},{:.4f},{:.4f}'.format(result_test[0], result_test[1], result_test[2], result_test[3]) +
                ' C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result_test[4], result_test[5], result_test[6], result_test[7]) +
                ' C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result_test[8], result_test[9], result_test[10], result_test[11]) +
                ' C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result_test[12], result_test[13], result_test[14], result_test[15]) +
                ' C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(result_test[16], result_test[17], result_test[18], result_test[19]))

# def train(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output = model(idx_train, train_idx)
#     loss_train = F.nll_loss(output, labels[idx_train])
#     acc_train = accuracy(output, labels[idx_train])
#     loss_train.backward()
#     optimizer.step()
#
#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output = model(idx_val, dev_idx)
#
#     val_output = model(idx_val, dev_idx)
#     loss_val = F.nll_loss(val_output, labels[idx_val])
#     acc_val = accuracy(val_output, labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.item()),
#           'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
#
# def test():
#     model.eval()
#     test_output = model(idx_test, test_idx)
#     loss_test = F.nll_loss(test_output, labels[idx_test])
#     acc_test = accuracy(test_output, labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))
#     print("=========================================")
#     y_pred = torch.max(test_output, dim=1)[1].data.cpu().numpy().tolist()
#     test_labels = labels[idx_test].data.cpu().numpy().tolist()
#     print(classification_report(test_labels, y_pred, target_names=args.target_names, digits=5))
#
#
# # Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Testing
# test()
