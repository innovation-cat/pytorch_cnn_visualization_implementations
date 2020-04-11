
import numpy as np

from models import get_model
from torchsummary import summary
import torch, utils, cv2
from torchvision import transforms
from torch.optim import Adam, SGD
from misc_functions import recreate_image, save_image
import matplotlib.pyplot as plt
import argparse, tqdm
import scipy.ndimage as nd

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
	
def dream(image, model, iterations, lr, filter=-1):
	print(type(image), image.shape)
	#Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
	image = torch.from_numpy(image).cuda()

	image.requires_grad_(True)
	#image = torch.from_numpy(image).requires_grad_(True).cuda()

	for i in range(iterations):
		model.zero_grad()
		if filter==-1:
			out = model(image)
		else:
			out = model(image)[0][filter]
			
		loss = out.norm()
		loss.backward()
		avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
		norm_lr = lr / avg_grad
		image.data += norm_lr * image.grad.data
		image.data = utils.clip(image.data)
		image.grad.data.zero_()
	return image.cpu().data.numpy()
	
	
def deep_dream(image, model, iterations, lr, octave_scale, num_octaves, filter):

	octaves = [image.detach().numpy()]
	print(type(octaves[-1]), octaves[-1].shape, np.mean(octaves[-1]))
	
	for _ in range(num_octaves - 1):
		octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))
		
		print(type(octaves[-1]), octaves[-1].shape, np.mean(octaves[-1]))


	detail = np.zeros_like(octaves[-1])
	for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
		if octave > 0:
			# Upsample detail to new octave dimension
			detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
		# Add deep dream detail from previous octave to new base
		input_image = octave_base + detail
		# Get new deep dream image
		#dreamed_image = dream(input_image, model, iterations, lr)   #  all channel
		dreamed_image = dream(input_image, model, iterations, lr, filter)    # one channel
		# Extract deep dream details
		detail = dreamed_image - octave_base

	
	dreamed_image = dreamed_image.squeeze().transpose(1, 2, 0)
	dreamed_image = dreamed_image * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
	dreamed_image = np.clip(dreamed_image, 0.0, 255.0)

	utils.show_image(dreamed_image)
	
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_image", type=str, default="images/supermarket.jpg", help="path to input image")
	parser.add_argument("--iterations", default=20, help="number of gradient ascent steps per octave")
	parser.add_argument("--at_layer", default=34, type=int, help="layer at which we modify image to maximize outputs")
	parser.add_argument("--at_filter", default=45, type=int, help="filter of at_layer at which we modify image to maximize outputs")
	parser.add_argument("--lr", default=0.01, help="learning rate")
	parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
	parser.add_argument("--num_octaves", default=10, help="number of octaves")
	args = parser.parse_args()

	examples = [('dog.png', 263), ('elephant.jpg', 101)]
	
	idx = 1
	
	img_path = examples[idx][0]
	img_label = examples[idx][1]

	img = cv2.imread(img_path)

	print(np.mean(img))

	prep_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])(img)
	prep_img = torch.unsqueeze(prep_img, 0)
	
	model = get_model('vgg19')

	
	if "features" in dict(list(model.named_children())):
		conv_model = torch.nn.Sequential(*list(model.features.children())[:args.at_layer+1])  
	else:
		conv_model = torch.nn.Sequential(*list(model.children())[:args.at_layer+1])  
		

	deep_dream(prep_img, conv_model, args.iterations, args.lr, args.octave_scale, args.num_octaves, -1)
		
		
		