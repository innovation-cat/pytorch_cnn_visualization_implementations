# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def read_img(img_path, trans, size):
    """util function to read and preprocess the test image.
    Args:
           img_path: path of image.
           preprocess_input: preprocess_input function.
           size: resize.
    Returns:
           img: original image.
           pimg: processed image.
    """
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, size)

    pimg = np.expand_dims(pimg, axis=0)
    pimg = preprocess_input(pimg)

    return transforms.ToTensor()


def preprocess_image(pil_im):

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	im_as_arr = np.float32(pil_im)

	# Normalize the channels
	for channel, _ in enumerate(im_as_arr):
		im_as_arr[channel] /= 255
		im_as_arr[channel] -= mean[channel]
		im_as_arr[channel] /= std[channel]
	# Convert to float tensor
	im_as_ten = torch.from_numpy(im_as_arr).float().cuda()
	# Add one more channel to the beginning. Tensor shape = 1,3,224,224
	im_as_ten.unsqueeze_(0)
	# Convert to Pytorch variable
	im_as_ten.requires_grad_(True)
	#print(im_as_ten.dtype)
	return im_as_ten

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


def vis_conv(images, rows, cols, name, save_name):
	print(images.shape)
	for i in range(rows):
		for j in range(cols):
			
			filter_img = images[i*cols+j, ...]
			
			if name == "filter":
				if filter_img.shape[0]==3:
					filter_img = np.transpose(filter_img, (1, 2, 0))
				else:
					filter_img = filter_img[0]
			plt.subplot(rows, cols, i*cols+j+1)
			plt.imshow(filter_img)
			frame = plt.gca()
			frame.axes.get_yaxis().set_visible(False)
			frame.axes.get_xaxis().set_visible(False)


	plt.savefig('images\{}.jpg'.format(save_name), dpi=600)
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