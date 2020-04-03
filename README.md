# Pytorch_cnn_visualization_implementations
This repository including most of cnn visualizations techniques using pytorch


## feature map visualization

In this situation, we can directly visualize intermediate feature map via one forward pass. In the following illustrations, we use pretrained vgg16 model, and output layer_1, layer_5, layer_15, layer_30 respectively.

<div align='center'><img src="./images/feature_map_1.jpg" width="400"/><img src="./images/feature_map_6.jpg" width="400"/></div>
<div align='center'><img src="./images/feature_map_15.jpg" width="400"/><img src="./images/feature_map_29.jpg" width="400"/></div>


## convolutional filter visualization

We can directly visualize convolutional filter parameters as below.

This method is suitable for the first convolutional kernel, the results show that the first layer can learn simple features, such as edge, colored blobs. We can visualize filters at higher layers, but not that interesting.

<table border=0>
	<tbody>
		<tr>
			<td width="5%" align="center"> AlexNet </td>
			<td width="45%" > <img src="./images/alexnet_filter_0.jpg"> </td>
			<td width="45%"> <img src="./images/alexnet_filter_6.jpg"> </td>
		</tr>
        <tr>
			<td width="5%" align="center"> ResNet50 </td>
			<td width="45%" > <img src="./images/resnet50_filter_0.jpg"> </td>
			<td width="45%"> <img src="./images/resnet50_filter_18.jpg"> </td>
		</tr>
    </tbody>
</table>

