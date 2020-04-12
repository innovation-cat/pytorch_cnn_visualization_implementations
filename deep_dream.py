
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def gradient_ascent_intermediate_layer(prep_img, select_layer, select_filter=-1):

	model = get_model('vgg19')
	
	
	if "features" in dict(list(model.named_children())):
		conv_model = torch.nn.Sequential(*list(model.features.children())[:select_layer+1])  
	else:
		conv_model = torch.nn.Sequential(*list(model.children())[:select_layer+1])  

	
	optimizer = Adam([prep_img], lr=0.1, weight_decay=1e-6)	
	for i in range(1, 201):
		optimizer.zero_grad()
		
		if select_filter == -1:
			output = conv_model(prep_img)[0]
		else:
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

	examples = [('dog.png', 263), ('elephant.jpg', 101), ('Borderllie.jpg', 231)]
	
	idx = 0
	
	img_path = examples[idx][0]
	img_label = examples[idx][1]

	img = cv2.imread(img_path)

	prep_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])(img)
	
	prep_img = torch.unsqueeze(prep_img, 0).cuda()
	
	prep_img.requires_grad_(True)
	

	
	select_layer, select_filter = 34, 45 
	
	gradient_ascent_intermediate_layer(prep_img, select_layer, select_filter)
