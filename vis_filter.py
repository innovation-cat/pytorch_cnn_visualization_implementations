# -*- coding: utf-8 -*-
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms


if __name__ == '__main__':

	model = get_model('densenet121')  # change model what you want
	
	select_layer = 26
	
	for i, (name, p) in enumerate(model.named_parameters()):
		if i==select_layer:
			output = p 
			break

	
	utils.vis_conv(output.cpu().detach().numpy(), 5, 5, "filter", "%s_filter_%d"%("densenet121", select_layer))	

	