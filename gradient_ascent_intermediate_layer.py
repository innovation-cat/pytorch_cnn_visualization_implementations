
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

def gradient_ascent_intermediate_layer(prep_img, select_layer, select_filter):

	model = get_model('vgg16')
	
	
	if "features" in dict(list(model.named_children())):
		conv_model = torch.nn.Sequential(*list(model.features.children())[:select_layer+1])  
	else:
		conv_model = torch.nn.Sequential(*list(model.children())[:select_layer+1])  

	
	optimizer = Adam([prep_img], lr=0.1, weight_decay=1e-6)	
	for i in range(1, 201):
		optimizer.zero_grad()
		
		output = conv_model(prep_img)[0][select_filter]
		
		loss = -torch.mean(output)
		print(i, "->", loss)
		
		loss.backward()
		
		optimizer.step()
		
		created_image = utils.recreate_image(prep_img)
		
		if i % 5 == 0:
			im_path = '../generated/layer_vis_%d.jpg'%i
			utils.save_image(created_image, im_path)


if __name__ == "__main__":

	random_image = np.uint8(np.random.uniform(140, 180, (400, 600, 3)))
	prep_img = transforms.ToTensor()(random_image)
	
	prep_img = torch.unsqueeze(prep_img, 0).cuda()
	
	prep_img.requires_grad_(True)
	

	
	select_layer, select_filter = 24, 25 
	
	gradient_ascent_intermediate_layer(prep_img, select_layer, select_filter)

		
		
		