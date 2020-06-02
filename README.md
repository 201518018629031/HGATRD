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

These dastasets are preprocessed according to our requirement and original datasets can be available at [here](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0)

## Training & Testing
1. create an empty directory: result/
2. sh run.sh 0 twitter15\[twitter16\]

# Citation
If you find the code is useful for your research, please cite this paper:  
<pre><code>@inproceedings{inproceedings,
author = {Huang, Qi and Yu, Junshuai and Wu, Jia and Wang, Bin},
year = {2020},
month = {06},
pages = {1-8},
title = {Heterogeneous Graph Attention Networks for Early Detection of Rumors on Twitter},
doi = {}
}</code></pre>

