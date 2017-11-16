## Project Writeup: Follow Me using Deep Learning

### Objective

This project consists of designing and training a deep neural network, so that a quadrotor can follow a specific person through a crowd.

### Approach

The deep neural network to be designed should be capable of scene understanding, i.e., to be able to do pixel-wise classification.

For this, the neural network needs to have convolution layers, in order to perceive the objects in the input. Three of these layers (A, B, C) are connected, where A is connected to B, and this one to C. The A layer receives the input. The width and height of each layer is reduced by half with respect to its previous layer. Hence, the width and height of A is twice the with and height of B, respectively. And the same of B with respect to C.

The reason to have convolution layers, is to be able to group adjacent pixels and characterize features present on such pixels based on this adjacency. This characterization is achieved by sharing a set of parameters across the image. Because they are shared, a determined type of features can be characterized, e.g., lines with a determined inclination, curves. The objective of training a convolutional network is to learn these shared parameters.

The reason to connect three convolution layers is to characterize features hierarchically, i.e., the third layer can characterize features made of features that are characterized by the second layer, and this one in turn can learn to characterize features composed of features characterized by the first layer. For example, the first layer can characterize curves and lines. Then the second layer can characterize circles and ovals, both of them made of curves. And the third layer can characterize an eye, which is made of circles, lines, ovals, among other features.

In the case of this project, the target to be identified has enough complexity, in terms of the features it is made of, in order to require at least three convolution layers.

Each of the outputs of these first three convolution layers is normalized, in order to avoid introducing errors when values of very different magnitudes are operated, e.g., a very big value plus a tiny small value. Additionally, convergence is quicker using normalized values.

The output of the third convolution layer is fed to a 1x1 convolution layer, i.e., one that has a 1x1 kernel. The purpose of this is to avoid losing dimensional information.

The output of the 1x1 convolution layer is up-sampled until the size of the original input images is obtained. The purpose of up-sampling is to be able to do pixel-wise classification. In order to obtain the original size, the same number of convolution layers (three) are added to the network after the 1x1 convolution layer. These three up-sampling layers are called transposed convolution layers, because they reverse the convolution operation performed in the first three layers.

Additionally, a technique called skip connections is used in order to be able to do precise pixel-wise classification. This is possible because skip connections allows the network to recover feature localization information that was lost while the images where convoluted in the first three layers. The way it works is by connecting the output of a convolution layer.

Finally, a pair of convolution layers are added to the end of the transposed convolution layers with skip-connections, in order to extract some more spatial information.

### Architecture

[//]: # (Image References)

[image1]: ./misc_images/robond-kinematics.jpg

The following figure depicts the arquitecture of the fully convolutional network used:

![alt text][image1]

The inputs layer represents the input to the system. The block1 layer is the result of applying a 3x3 convolution with same padding, a stride of 2x2, which reduces the size of the output to a half with respect to the input, and relu activation. The depth of the output is increased to 32 channels. This output is batch-normalized.

In a like manner, the block2 and block3 are obtained from 3x3 convolutions with same padding, stride of 2x2, and relu activation. But the depth of block2 and block3 are set to 64 channels and 128 channels, respectively.

The stride of 2x2 implies that block2 width and height have half the width and height of block1, and that block3 width and height have half the width and height of block2.

From inputs to block3, it can be noted that the size reduces in half, and the depth increases twice, except in the case of block1 with respect to inputs.

conv_layer is the result of applying a 1x1 convolution with same padding and a stride of 1x1, which makes conv_layer to have the same width and height of block3, but a chosen depth of 64. This result is batch normalized as well.

The conv_layer is bilinearly up-sampled to twice its size to obtain the up-sampled4 layer. This one in turn is concatenated to the block2 layer in order to produce a skip-connection.

A 3x3 separable convolution with same padding, a stride of 1x1 and a rule activation is applied to the skip-connection, and the result is batch normalized. Then this output is applied again an exactly equal process, to obtain the block4 layer.

In the same manner, the block4/5 layer is applied the same set of convolutions and skip-connection as the ones used in the conv_layer to obtain the block4 layer, but in this case the layer obtained is the block5/x.

Finally, the x layer is applied a 3x3 convolution with same padding, a stride of 1x1 and softmax activation, and this result is batch normalized.

