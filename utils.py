# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch, copy, os
from torchvision import transforms
from PIL import Image
import matplotlib.cm as mpl_color_map

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def preprocess_image(image):
	prep_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])(image)
	return prep_img

def recreate_image(im_as_var):
	"""
		Recreates images from a torch variable, sort of reverse preprocessing
	Args:
		im_as_var (torch variable): Image to recreate
	returns:
		recreated_im (numpy arr): Recreated image in array
	"""
	reverse_mean = [-0.485, -0.456, -0.406]
	reverse_std = [1/0.229, 1/0.224, 1/0.225]
	if isinstance(im_as_var, torch.Tensor):
		print("is tensor")
		recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0])
	else:
		print("is array")
		recreated_im = copy.copy(im_as_var[0])
		
	for c in range(3):
		recreated_im[c] /= reverse_std[c]
		recreated_im[c] -= reverse_mean[c]
	recreated_im[recreated_im > 1] = 1
	recreated_im[recreated_im < 0] = 0
	recreated_im = np.round(recreated_im * 255)

	recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
	return recreated_im



def vis_conv(images, rows, cols, name, save_name):
	print(np.max(images), np.min(images), images.shape)
	#plt.suptitle(save_name)
	figure = plt.figure()
	for i in range(rows):
		for j in range(cols):
			filter_img = images[i*cols+j, ...]
			if name == "filter":
				if filter_img.shape[0]==3:
					filter_img = np.transpose(filter_img, (1, 2, 0))
				else:
					filter_img = filter_img[0]  # only show the first filter
			plt.subplot(rows, cols, i*cols+j+1)
			plt.imshow(filter_img)
			frame = plt.gca()
			frame.axes.get_yaxis().set_visible(False)
			frame.axes.get_xaxis().set_visible(False)

	figure.suptitle(save_name, fontsize=18)
	#plt.savefig('outputs\{}.jpg'.format(save_name), dpi=600)
	plt.show()

def show_color_gradients(gradients, save_name):
	# normalize 
	gradients = gradients - gradients.min()
	gradients /= gradients.max()
	gradients = (gradients*255).astype(np.uint8)	
	gradients = np.transpose(gradients, (1, 2, 0))
	show_image(gradients)


def convert_to_grayscale(im_as_arr):
	grayscale_im = np.max(np.abs(im_as_arr), axis=0)
	#im_max = np.max(grayscale_im)
	im_max = np.percentile(grayscale_im, 99)
	im_min = np.min(grayscale_im)
	grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
	grayscale_im = np.expand_dims(grayscale_im, axis=0)
	return grayscale_im
	
	
def show_gray_gradients(gradients, save_name):
	gradients = convert_to_grayscale(gradients)
	gradients = np.repeat(gradients, 3, axis=0)
	gradients = np.transpose(gradients, (1, 2, 0))
	
	show_image(gradients)
	
	
def show_image(image):
	frame = plt.gca()
	frame.axes.get_yaxis().set_visible(False)
	frame.axes.get_xaxis().set_visible(False)
	plt.imshow(image)
	#plt.savefig('outputs\{}.jpg'.format(save_name), dpi=600)
	plt.show()
	
	
def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def clip(image_tensor):

	for c in range(3):
		m, s = mean[c], std[c]
		image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
	return image_tensor

def save_image(im, path):
	"""
		Saves a numpy matrix or PIL image as an image
	Args:
		im_as_arr (Numpy array): Matrix of shape DxWxH
		path (str): Path to the image
	"""
	if isinstance(im, (np.ndarray, np.generic)):
		im = format_np_output(im)
		print(im.shape)
		im = Image.fromarray(im)
	im.save(path)	
	
	
def save_class_activation_images(org_img, activation_map, file_name):
	"""
		Saves cam activation map and activation map on the original image

	Args:
		org_img (PIL img): Original image
		activation_map (numpy arr): Activation map (grayscale) 0-255
		file_name (str): File name of the exported image
	"""
	if not os.path.exists('../results'):
		os.makedirs('../results')
	# Grayscale activation map
	heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
	# Save colored heatmap
	path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
	save_image(heatmap, path_to_file)
	# Save heatmap on iamge
	path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
	save_image(heatmap_on_image, path_to_file)
	# SAve grayscale heatmap
	path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
	save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
	"""
		Apply heatmap on image
	Args:
		org_img (PIL img): Original image
		activation_map (numpy arr): Activation map (grayscale) 0-255
		colormap_name (str): Name of the colormap
	"""
	# Get colormap
	color_map = mpl_color_map.get_cmap(colormap_name)
	no_trans_heatmap = color_map(activation)
	# Change alpha channel in colormap to make sure original image is displayed
	heatmap = copy.copy(no_trans_heatmap)
	heatmap[:, :, 3] = 0.4
	heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
	no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

	# Apply heatmap on iamge
	heatmap_on_image = Image.new("RGBA", org_im.size)
	heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
	
	#heatmap = heatmap.transpose(method=Image.TRANSVERSE)
	print(heatmap_on_image)
	print(heatmap)
	
	heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
	return no_trans_heatmap, heatmap_on_image