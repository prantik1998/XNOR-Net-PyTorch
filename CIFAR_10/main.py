from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util

import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

import click
import yaml

from pipelinemanager import pipelinemanager

@click.group()
def main():
	pass

@main.command()
def train():
	manager.train(trainloader)

@main.command()
def test():
	manager.test(testloader)



if __name__=='__main__':
	config=yaml.safe_load(open("config/config.yaml","r"))
	manager=pipelinemanager(torch.cuda.is_available(),config)

	# # set the seed
	# torch.manual_seed(1)
	# torch.cuda.manual_seed(1)

	# # prepare the data
	# if not os.path.isfile(args.data+'/train_data'):
	# 	# check the data path
	# 	raise Exception\
	# 			('Please assign the correct data path with --data <DATA_PATH>')

	trainset = data.dataset(root=config["dir"], train=True)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batchsize"],
			shuffle=True, num_workers=2)

	testset = data.dataset(root=config["dir"], train=False)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100,
			shuffle=False, num_workers=2)

	# define classes
	classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	main()

	# define the model
	# if args.arch == 'nin':
	# 	model = nin.Net()
	

	# # initialize the model
	# if not args.pretrained:
	# 	print('==> Initializing model parameters ...')
	# 	best_acc = 0
	# 	for m in model.modules():
	# 		if isinstance(m, nn.Conv2d):
	# 			m.weight.data.normal_(0, 0.05)
	# 			m.bias.data.zero_()
	# else:
	# 	print('==> Load pretrained model form', args.pretrained, '...')
	# 	pretrained_model = torch.load(args.pretrained)
	# 	best_acc = pretrained_model['best_acc']
	# 	model.load_state_dict(pretrained_model['state_dict'])

	# if not args.cpu:
	# 	model.cuda()
	# 	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
	# print(model)

	# # define solver and criterion
	# base_lr = float(args.lr)
	# param_dict = dict(model.named_parameters())
	# params = []



	
	# define the binarization operator

	# do the evaluation if specified
