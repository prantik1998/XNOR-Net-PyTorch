import numpy as np 
import os 
import util
import sys


import torch
# from test import test,test_folder
# from train import train
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from models import nin


def save_state(model,modelpath,best_acc,epoch):
	print('==> Saving model ...')
	state = {
			'best_acc': best_acc,
			'state_dict': model.state_dict(),
			}
	for key in state['state_dict'].keys():
		if 'module' in key:
			state['state_dict'][key.replace('module.', '')] = \
					state['state_dict'].pop(key)
	torch.save(state,modelpath+str(epoch)+'.pth.tar')

def train(model,trainloader,binop,lr,epochs,modelpath):
	best_acc = 0
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()
	for epoch in range(epochs):
		correct = 0
		for batch_idx, (data, target) in enumerate(trainloader):
			# process the weights including binarization
			binop.binarization()
			
			# forwarding
			data, target = Variable(data.cuda()), Variable(target.cuda())
			optimizer.zero_grad()
			output = model(data)
			
			# backwarding
			loss = criterion(output, target.long())
			loss.backward()
			
			# restore weights
			binop.restore()
			binop.updateBinaryGradWeight()
			
			optimizer.step()
			if batch_idx % 100 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
					epoch, batch_idx * len(data), len(trainloader.dataset),
					100. * batch_idx / len(trainloader), loss.data.item(),
					optimizer.param_groups[0]['lr']))

			pred = output.data.max(1, keepdim=True)[1]
			pred = pred.squeeze()
			correct += pred.eq(target.long()).cpu().sum()

		acc = 100. * float(correct) / len(trainloader.dataset)
		print(f'Accuracy:-{acc},Best_Accuracy:-{best_acc}')
		if acc > best_acc:
			best_acc = acc
			save_state(model,modelpath,best_acc,epoch)



# def adjust_learning_rate(optimizer, epoch):
# 	update_list = [120, 200, 240, 280]
# 	if epoch in update_list:
# 		for param_group in optimizer.param_groups:
# 			param_group['lr'] = param_group['lr'] * 0.1
# 	return


def test(model,binop,testloader):
	model.eval()
	correct = 0
	binop.binarization()
	for data, target in testloader:
		data, target = Variable(data.cuda()), Variable(target.cuda())									
		output = model(data)
		pred = output.data.max(1, keepdim=True)[1]
		correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()
	binop.restore()
	acc = 100. * float(correct) / len(testloader.dataset)

	
	print(f"Accuracy:-{acc}")


class pipelinemanager:
	def __init__(self,cuda,config):
		self.config = config
		self.device = torch.device("cuda" if cuda else "gpu")
		self.model = nin.Net().to(self.device)
		self.binop = util.BinOp(self.model)
		if "pretrained" in self.config.keys():
			print("Loading pretrained model")
			self.model.load_state_dict(torch.load(self.config["pretrained"])['state_dict'])



	def train(self,trainloader):
		train(self.model,trainloader,self.binop,self.config["lr"],self.config["epochs"],self.config["modelpath"])

	def test(self,testloader):
		test(self.model,self.binop,testloader)

	def test_folder(self,testloader):
		test_folder(self.binop,self.config["test_folder"],testloader)