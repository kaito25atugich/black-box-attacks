import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utils
from oracle import oracle
from predict import predict
from model import Classifier
import torch.utils.data as TorchUtils

def train(model, epochs, dataset, test_dataset, criterion, optimizer, device):
	
	BATCH_SIZE = 100

	for epoch in range(1, epochs+1):
		train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

		# Update (Train)
		model.train()

		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data.to(device)), Variable((target).to(device))

			optimizer.zero_grad()
			model.zero_grad()

			output = model(data)
			loss = criterion(output, target)

			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()

			loss.backward()
			optimizer.step()

		predict(model, test_dataset, device)


def create_dataset(dataset, labels): 
	# 
	# for img in dataset:
	# 	print(len(img))

	data = torch.stack([img[0] for img in dataset])
	target = torch.stack([label[0] for label in labels])

	new_dataset = TorchUtils.TensorDataset(data,target)
	return new_dataset

	
def augment_dataset(model, dataset, LAMBDA, device):

	new_dataset = list()
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

	for data, target in data_loader:

		data, target = Variable(data.to(device), requires_grad=True), Variable(target.to(device))
		model.zero_grad()

		output = model(data)
		output[0][target].backward()

		data_new = data[0] + LAMBDA * torch.sign(data.grad.data[0])
		new_dataset.append(data_new.cpu())

	new_dataset = torch.stack([data_point for data_point in new_dataset])
	new_dataset = TorchUtils.TensorDataset(new_dataset)

	# new_dataset = TorchUtils.ConcatDataset([dataset, new_dataset])
	return new_dataset



def train_substitute(oracle_model, dataset, test_dataset, device, MAX_RHO, LAMBDA, EPOCHS): 

	oracle_model = oracle_model.to(device)

	model = None
	for rho in range(MAX_RHO):
		print(len(dataset))

		dummy_labels = oracle(oracle_model, dataset, device)
		dummy_dataset = create_dataset(dataset, dummy_labels)

		model = Classifier().to(device)
		criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.Adagrad(model.parameters(), lr=0.01)

		train(model, EPOCHS, dummy_dataset, test_dataset, criterion, optimizer, device)
		print("Rho: %d"%(rho))
		print("Dataset Size: %d"%(len(dataset)))

		dataset = augment_dataset(model, dummy_dataset, LAMBDA, device)

	return model

def train_substitute_not_scratch(oracle_model, dataset, test_dataset, device, MAX_RHO, LAMBDA, EPOCHS): 

	oracle_model = oracle_model.to(device)

	model = Classifier().to(device)
	for rho in range(MAX_RHO):

		dummy_labels = oracle(oracle_model, dataset, device)
		dummy_dataset = create_dataset(dataset, dummy_labels)

		criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.Adagrad(model.parameters(), lr=0.01)

		train(model, EPOCHS, dummy_dataset, test_dataset, criterion, optimizer, device)
		print("Rho: %d"%(rho))
		print("Dataset Size: %d"%(len(dataset)))

		dataset = augment_dataset(model, dummy_dataset, LAMBDA, device)

	return model
