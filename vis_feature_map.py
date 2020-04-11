# -*- coding: utf-8 -*-
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms



if __name__ == '__main__':

	img_path = './inputs/dog.png'

	img = cv2.imread(img_path)

	input = transforms.ToTensor()(img)
	
	input = torch.unsqueeze(input, 0).cuda()

	
	model = get_model('vgg16')
	
	select_layer = 1
	
	if "features" in dict(list(model.named_children())):
		conv_model = torch.nn.Sequential(*list(model.features.children())[:select_layer+1])  
	else:
		conv_model = torch.nn.Sequential(*list(model.children())[:select_layer+1])  


	conv_model.cuda()
	output = conv_model(input)[0]

	
	utils.vis_conv(output.cpu().detach().numpy(), 8, 8, "conv", "feature_map_%d"%select_layer)

	