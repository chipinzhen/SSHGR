# A-Self-Supervised-Based-Hierarchical-Graph-Representation-Learning-for-DDI-Prediction

The pretraining code is adapted from `pretrain-gnns` repo: https://github.com/Dinxin/pretrain-gnns

And we also followed the DDI network (GCN) setting from `MIRACLE` repo: https://github.com/isjakewong/MIRACLE

## Installation
We used the conda environment with several python packagess.
The details of environment and python packages we used has been put in the file "environment.yaml" and "requirements.txt".

## Dataset download
In the pre-traning with SSL, we use a dataset named ZINC15 which can be downloaded from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip)
The dataset of DDI task have been provided.




# Reference
D. E. Shaw, J. Grossman, J. A. Bank, B. Batson, J. A. Butts, J. C. Chao, M. M. Deneroff, R. O. Dror, A. Even, C. H. Fenton, et al. Anton 2: raising the bar for performance and programmability in a special-purpose molecular dynamics supercomputer. In SC’14: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 41–53. IEEE, 2014

W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec. Strategies for pre-training graph neural networks. arXiv preprint arXiv:1905.12265, 2019.

Y. Wang, Y. Min, X. Chen, and J. Wu. Multi-view graph contrastive representation learning for drug-drug interaction prediction. In Proceedings of the Web Conference 2021, pages 2921–2933, 2021.
