# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch, copy
from torchvision import transforms
from PIL import Image

def deprocess_image(x):
    """util function to convert a tensor into a valid image.
    Args:
           x: tensor of filter.
    Returns:
           x: deprocessed tensor.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #if K.image_data_format() == 'channels_first':
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def normalize(x):
    """utility function to normalize a tensor by its L2 norm
    Args:
           x: gradient.
    Returns:
           x: gradient.
    """
    return x / (torch.sqrt(torch.mean(torch.mul(x, x))) + 1e-07)

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
    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0])
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
	plt.savefig('images\{}.jpg'.format(save_name), dpi=600)
	plt.show()

def show_color_gradients(gradients, save_name):
	# normalize 
	gradients = gradients - gradients.min()
	gradients /= gradients.max()
	gradients = (gradients*255).astype(np.uint8)	
	gradients = np.transpose(gradients, (1, 2, 0))
	frame = plt.gca()
	frame.axes.get_yaxis().set_visible(False)
	frame.axes.get_xaxis().set_visible(False)
	plt.imshow(gradients)
	plt.show()


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
	frame = plt.gca()
	frame.axes.get_yaxis().set_visible(False)
	frame.axes.get_xaxis().set_visible(False)
	plt.imshow(gradients)
	#plt.savefig('images\{}.jpg'.format(save_name), dpi=600)
	plt.show()
	
def vis_heatmap(img, heatmap):
	"""visualize heatmap.
	Args:
		   img: original image.
		   heatmap：heatmap.
	"""
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	plt.figure()

	plt.subplot(221)
	plt.imshow(cv2.resize(img, (224, 224)))
	plt.axis('off')

	plt.subplot(222)
	plt.imshow(heatmap)
	plt.axis('off')

	plt.subplot(212)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap)
	# We apply the heatmap to the original image
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = np.uint8(heatmap * 0.4 + img)
	superimposed_img = superimposed_img[...,::-1]
	plt.imshow(superimposed_img)
	plt.axis('off')

	plt.tight_layout()
	plt.savefig('images\heatmap.jpg', dpi=600)
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