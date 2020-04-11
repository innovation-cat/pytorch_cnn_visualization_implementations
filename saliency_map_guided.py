
# -*- coding: utf-8 -*-
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms
from misc_functions import *
import matplotlib.pyplot as plt
from torch.nn import ReLU


class GuidedBackprop():
	def __init__(self, model):
		self.model = model
		self.gradients = None 
		self.model.eval()
		self.relu_activations = []
		self.adjust_relu_layer()
		self.hook_layer()
		
		
	def adjust_relu_layer(self):
		def foreward_fn(module, grad_in, grad_out):
			self.relu_activations.append(grad_out[0])
			
		def backward_fn(module, grad_in, grad_out):
			grad = self.relu_activations.pop()
			grad[grad > 0] = 1
			output = grad * torch.clamp(grad_out[0], min=0.0)
			return (output, )
		
		for module in self.model.features.children():
			if isinstance(module, ReLU):
				module.register_forward_hook(foreward_fn)
				module.register_backward_hook(backward_fn)
		
	def hook_layer(self):
		def hook_fn(module, grad_in, grad_out):
			# grad_in: (input, weights, bias)
			self.gradients = grad_in[0]
			
		if "features" in dict(list(self.model.named_children())):
			first_layer = list(self.model.features.children())[0]
		else:
			first_layer = list(self.model.children())[0]
			
		first_layer.register_backward_hook(hook_fn)	
		
	def generate_gradients(self, model_input, target_class):
		
		model_output = self.model(model_input)
		
		print(model_output.data.max(1)[1].item(), target_class)
		#assert(model_output.data.max(1)[1].item() == target_class)
		
		self.model.zero_grad()
		mask = torch.zeros(model_output.size(), dtype=torch.float32).cuda()
		mask[0][target_class] = 1.0
		
		model_output.backward(gradient=mask)
			
		gradients_as_arr = self.gradients.data.cpu().numpy()[0]
		return gradients_as_arr	


if __name__ == "__main__":	
	
	examples = [('./inputs/dog.png', 263), ('./inputs/elephant.jpg', 101)]
	
	idx = 1
	
	img_path = examples[idx][0]
	img_label = examples[idx][1]

	img = cv2.imread(img_path)

	print(img.shape)
	prep_img = transforms.ToTensor()(img)
	
	prep_img = torch.unsqueeze(prep_img, 0).cuda()
	
	prep_img.requires_grad_(True)

	model = get_model('vgg16')

	VBP = GuidedBackprop(model)
	gradients = VBP.generate_gradients(prep_img, img_label)

	utils.show_color_gradients(gradients, "%s_guided_backpropagation_color"%img_path.split('.')[0])

	utils.show_gray_gradients(gradients, "%s_guided_backpropagation_gray"%img_path.split('.')[0])



