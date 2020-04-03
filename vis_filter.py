# -*- coding: utf-8 -*-
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms


if __name__ == '__main__':

	img_path = 'dog.png'

	img = cv2.imread(img_path)

	input = transforms.ToTensor()(img)
	
	input = torch.unsqueeze(input, 0).cuda()

	model = get_model('alexnet')
	summary(model, (3, 244, 244))
	
	select_layer = 6
	
	for i, (name, p) in enumerate(model.named_parameters()):
		if i==select_layer:
			output = p 
			break
	
	utils.vis_conv(output.cpu().detach().numpy(), 8, 8, "filter", "alexnet_filter_%d"%select_layer)	

	