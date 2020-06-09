# HGATRD
The implementation of our IJCNN 2020 paper "Heterogeneous Graph Attention Networks for Rumor Detection on Twitter" 
# Requirements
python 3.6.6  
numpy==1.17.2  
scipy==1.3.1  
pytorch==1.1.0  
scikit-learn==0.21.3  
# How to use
## Dataset
The main directory contains the directories of two Twitter datasets: twitter15 and twitter16. In each directory, there are:  
ind.twitter1*.label files: the labels of all nodes in the tweet-word subgraph.  
ind.twitter1*.features_index files: the index of words that nodes in the tweet-word subgraph contained.  
ind.twitter1*.adj files: the weight adjacency matrix of the tweet-word subgraph.  
ind.twitter1*.train files: the index of the training set in the tweet-word subgraph.  
ind.twitter1*.dev files: the index of the validation set in the tweet-word subgraph.  
ind.twitter1*.test files: the index of the testing set in the tweet-word subgraph.  
ind.twitter1*.user.tweet.adj files: the weight adjacency matrix of the tweet-user subgraph.  
ind.twitter1*.user.tweet.train files: the index of the training set in the tweet-user subgraph.  
ind.twitter1*.user.tweet.dev files: the index of the validation set in the tweet-user subgraph.  
ind.twitter1*.user.tweet.test files: the index of the testing set in the tweet-user subgraph.  
ind.twitter1*.user.tweet.adj.tc* files: the weight adjacency matrix of the tweet-user subgraph which simulated the task of early detection by controlling the received tweets.  
ind.twitter1*.user.tweet.adj.et* files: the weight adjacency matrix of the tweet-user subgraph which simulated the task of early detection by controlling the elapsed time.  
These datasets are preprocessed according to our requirement and original datasets can be available at [here](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0)

## Training & Testing
sh run.sh 0 twitter15\[twitter16\]

# Citation
If you find the code is useful for your research, please cite this paper:  
<pre><code>@inproceedings{huang2020heterogeneous,
author = {Huang, Qi and Yu, Junshuai and Wu, Jia and Wang, Bin},
year = {2020},
month = {06},
pages = {1-8},
title = {Heterogeneous Graph Attention Networks for Early Detection of Rumors on Twitter},
doi = {}
}</code></pre>
