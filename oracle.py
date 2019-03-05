import torch
import utils

def oracle(data, model_name):

	device = utils.get_device(1)
	model = torch.load("saved_models/"+model_name).to(device)

	data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
	pred = None

	model.eval()
	with torch.no_grad():
		for data in data_loader:
				data = data[0].to(device)
				output = model(data)
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	
	return pred