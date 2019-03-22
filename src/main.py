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

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--b_sz', type=int, default=50)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--gcn', action='store_true',
					help='use CUDA')
parser.add_argument('--config', type=str, default='./src/experiments.conf')
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
	print(features.size())

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn)
	graphSage.to(device)

	num_labels = len(set(getattr(dataCenter, ds+'_labels')))
	classification = Classification(config['setting.hidden_emb_size'], num_labels)
	classification.to(device)

	for epoch in range(args.epochs):
		print('----------------------EPOCH %d-----------------------' % epoch)
		apply_model(dataCenter, ds, graphSage, classification, args.b_sz, device)
		evaluate(dataCenter, ds, graphSage, classification, args.b_sz, device)

