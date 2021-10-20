""" Use unsupervised graphSAGE to encode ICD nodes

"""
import sys
import os
import torch
import argparse
import pyhocon
import random

from src.dataCenter import *
from src.utils import *
from src.models import *

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='icd10')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--b_sz', type=int, default=20)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataSet
    dataCenter = DataCenter(config)
    dataCenter.load_dataSet(ds)
    features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)

    graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
    graphSage.to(device)

    
    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), 
                                        getattr(dataCenter, ds+'_train'), device)
    
    print('GraphSage with Net Unsupervised Learning')
    early_stop = 0
    min_loss = np.inf
    for epoch in range(args.epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        graphSage, loss = apply_unsup_training(dataCenter, ds, graphSage, unsupervised_loss, args.b_sz, args.unsup_loss, device)
        
        if loss < min_loss:
            min_loss = loss
            early_stop = 0
            torch.save(graphSage.state_dict(), f'./models/unsup_graphsage_epoch_{epoch}')
            print(f"Saved model state at epoch {epoch}: f'./models/unsup_graphsage_epoch_{epoch}")
        else:
            early_stop += 1

        if early_stop >= args.patience:
            break
    
    # TODO: Extract and save embeddings


