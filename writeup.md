## Project Writeup: Follow Me using Deep Learning

### Objective

This project consists of designing and training a deep neural network, so that a quadrotor can follow a specific person through a crowd.

### Approach

The deep neural network to be designed should be capable of scene understanding, i.e., to be able to do pixel-wise classification.

For this, the neural network needs to have convolution layers, in order to perceive the objects in the input. Three of these layers (A, B, C) are connected, where A is connected to B, and this one to C. The A layer receives the input. The width and height of each layer is reduced by half with respect to its previous layer. Hence, the width and height of A is twice the with and height of B, respectively. And the same of B with respect to C.

The reason to have convolution layers, is to be able to group adjacent pixels and characterize features present on such pixels based on this adjacency. This characterization is achieved by sharing a set of parameters across the image. Because they are shared, a determined type of features can be characterized, e.g., lines with a determined inclination, curves. The objective of training a convolutional network is to learn these shared parameters.

The reason to connect three convolution layers is to characterize features hierarchically, i.e., the third layer can characterize features made of features that are characterized by the second layer, and this one in turn can learn to characterize features composed of features characterized by the first layer. For example, the first layer can characterize curves and lines. Then the second layer can characterize circles and ovals, both of them made of curves. And the third layer can characterize an eye, which is made of circles, lines, ovals, among other features.

In the case of this project, the target to be identified has enough complexity, in terms of the features it is made of, in order to require at least three convolution layers.

The output of the convolution layers, are fed to a 1x1 convolution layer, i.e., one that has a 1x1 kernel. The idea is to avoid losing dimensional information.

The output of the 1x1 convolution layer is up-sampled until the size of the original input images is obtained. The purpose of up-sampling is to be able to do pixel-wise classification. In order to obtain the original size, the same number of convolution layers (three) are added to the network after the 1x1 convolution layer. These three up-sampling layers are called transposed convolution layers, because they reverse the convolution operation performed in the first three layers.

Additionally, a technique called skip connections is used in order to be able to do precise pixel-wise classification. This is possible because skip connections allows the network to recover feature localization information that was lost while the images where convoluted in the first three layers. The way it works is by connecting the ouput of a convolution layer

### Architecture
