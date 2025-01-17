## A PyTorch implementation of GraphSAGE for Medical Ontology embedding
[Lui Sheng Jie](httpsL//github.com/luishengjie) (luishengjie@outlook.com; luishengjie@u.nus.edu)


## Forked from: A PyTorch implementation of GraphSAGE

- This package contains a PyTorch implementation of [GraphSAGE](http://snap.stanford.edu/graphsage/).

- This package is a fork of the GraphSage implementation by
[Tianwen Jiang](https://github.com/twjiang) (tjiang2@nd.edu), 
[Tong Zhao](https://github.com/zhao-tong) (tzhao2@nd.edu),
[Daheng Wang](https://github.com/adamwang0705) (dwang8@nd.edu).


## Environment settings

- python==3.6.8
- pytorch==1.0.0


## Basic Usage

**Main Parameters:**

```
--dataSet     The input graph dataset. (default: cora)
--agg_func    The aggregate function. (default: Mean aggregater)
--epochs      Number of epochs. (default: 50)
--b_sz        Batch size. (default: 20)
--seed        Random seed. (default: 824)
--unsup_loss  The loss function for unsupervised learning. ('margin' or 'normal', default: normal)
--config      Config file. (default: ./src/experiments.conf)
--cuda        Use GPU if declared.
```

**Learning Method**

The user can specify a learning method by --learn_method, 'sup' is for supervised learning, 'unsup' is for unsupervised learning, and 'plus_unsup' is for jointly learning the loss of supervised and unsupervised method.

**Example Usage**

To run the unsupervised model on Cuda:
```
python -m src.main --epochs 50 --cuda --learn_method unsup
```

## Generate ICD10 Graph
To generate the edgelist for the ICD10 ontology, run the following script:
```
python src/icd10_graph.py --outdir icd10-data

```
```
python -m src.main --epochs 3 --learn_method unsup --dataSet='icd10'                                                      
```
