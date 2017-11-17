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

The output of the third convolution layer is fed to a 1x1 convolution layer, i.e., one that has a 1x1 kernel. The purpose of this is to avoid losing dimensional information, as it happens when using a fully connected layer, which reduces dimensionality to 2D. Using a 1x1 convolution, the dimensionality is not lost, and the number of filters can be set to a specified value (64 in this project).

The output of the 1x1 convolution layer is up-sampled until the size of the original input images is obtained. The purpose of up-sampling is to be able to do pixel-wise classification. In order to obtain the original size, the same number of convolution layers (three) are added to the network after the 1x1 convolution layer. These three up-sampling layers are called transposed convolution layers, because they reverse the convolution operation performed in the first three layers.

Additionally, a technique called skip connections is used in order to be able to do precise pixel-wise classification. This is possible because skip connections allows the network to recover feature localization information that was lost while the images where convoluted in the first three layers. The way it works is by connecting the output of a convolution layer.

Finally, a pair of convolution layers are added to the end of the transposed convolution layers with skip-connections, in order to extract some more spatial information.

### Architecture

[//]: # (Image References)

[image1]: ./misc_images/FCN.png
[image2]: ./misc_images/training_epoch10.png
[image3]: ./data/sample_evaluation_data/following_images/images/0_run1cam1_00016.jpeg
[image4]: ./data/runs/following_images_run_1/0_run1cam1_00016_prediction.png
[image5]: ./data/sample_evaluation_data/patrol_non_targ/images/2_run2cam1_02978.jpeg
[image6]: ./data/runs/patrol_non_targ_run_1/2_run2cam1_02978_prediction.png
[image7]: ./data/sample_evaluation_data/patrol_with_targ/images/2_run2cam1_03598.jpeg
[image8]: ./data/runs/patrol_with_targ_run_1/2_run2cam1_03598_prediction.png
[image9]: ./data/sample_evaluation_data/patrol_with_targ/images/4_run1cam1_02363.jpeg
[image10]: ./data/runs/patrol_with_targ_run_1/4_run1cam1_02363_prediction.png
[image11]: ./data/sample_evaluation_data/patrol_with_targ/images/6_run5cam1_00004.jpeg
[image12]: ./data/sample_evaluation_data/patrol_with_targ/masks/6__mask_00004.png
[image13]: ./data/runs/patrol_with_targ_run_1/6_run5cam1_00004_prediction.png
[image14]: ./data/sample_evaluation_data/patrol_with_targ/images/6_run5cam1_00407.jpeg
[image15]: ./data/runs/patrol_with_targ_run_1/6_run5cam1_00407_prediction.png

The following figure depicts the arquitecture of the fully convolutional network used:

![alt text][image1]

The inputs layer represents the input to the system. The block1 layer is the result of applying a 3x3 convolution with same padding, a stride of 2x2, which reduces the size of the output to a half with respect to the input, and relu activation. The depth of the output is increased to 32 filters. This output is batch-normalized.

In a like manner, the block2 and block3 are obtained from 3x3 convolutions with same padding, stride of 2x2, and relu activation. But the depth of block2 and block3 are set to 64 filters and 128 filters, respectively.

The stride of 2x2 implies that block2 width and height have half the width and height of block1, and that block3 width and height have half the width and height of block2.

From inputs to block3, it can be noted that the size reduces in half, and the depth increases twice, except in the case of block1 with respect to inputs.

conv_layer is the result of applying a 1x1 convolution with same padding and a stride of 1x1, which makes conv_layer to have the same width and height of block3, but a chosen depth of 64. This result is batch normalized as well.

The conv_layer is bilinearly up-sampled to twice its size to obtain the up-sampled4 layer. This one in turn is concatenated to the block2 layer in order to produce a skip-connection.

A 3x3 separable convolution with same padding, a stride of 1x1, a rule activation, and depth of 128 filters is applied to the skip-connection, and the result is batch normalized. Then this output is applied again an exactly equal process, to obtain the block4 layer.

In the same manner, the block4/5 layer is also up-sampled, convoluted and concatenated exactly in the same way as in the case of the conv_layer, but in this case the layer obtained is the block5/x. The difference is that the depth of block5 and x are 64 and 32 filters, respectively.

Finally, the x layer is applied a 3x3 convolution with same padding, a stride of 1x1 and softmax activation. The depth of this final resultant layer is defined by the parameter num_classes, which is set to 3 for this project.

#### Code

The FCN architural model is implemented in python, in the ```fcn_model``` function shown as follows:

```python
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    block1 = encoder_block(inputs, filters=32, strides=2)
    block2 = encoder_block(block1, filters=64, strides=2)
    block3 = encoder_block(block2, filters=128, strides=2)

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(block3, filters=64, kernel_size=1, strides=1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    block4 = decoder_block(conv_layer, block2, filters=128)
    block5 = decoder_block(block4, block1, filters=64)
    x = decoder_block(block5, inputs, filters=32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

```

The variable names are exactly the ones used in the aforementioned description, so that the reader can understand the code by referring to the previous text. The implementation of the functions ```encoder_block``` and ```decoder_block``` are as follows:

```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    output_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([output_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    
    return output_layer
```

Finally, the ```conv2d_batchnorm```, ```separable_conv2d_batchnorm``` and ```bilinear_upsample``` are as follows:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

The implementation of the ```SeparableConv2DKeras``` and ```BilinearUpSampling2D``` classes can be found in the utils/separable_conv2d.py file. ```layers``` can be imported from ```tensorflow.contrib.keras.python.keras```.

### Hyperparameters

The neural networks need a set of parameters to be able to learn the weights and biases. These parameters are called hiperparameters, because the neural network uses them to learn another parameters, i.e., the weights and biases.

The learning rate is set to a small value, 0.01, to increase the chances of convergence. The number of epochs is 10, which indicates that the network is trained 10 times on the training set. ```steps_per_epoch```, which indicates the number of batches of training images used in 1 epoch, is set to 200. Although it is recommended to keep the constraint of steps per epoch equal to the training set size divided by the batch size, it was found that a ```batch_size``` of 64 provided the required 0.4 score, so there was no need to reduce it to a value of approximately 20.

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 10
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

```validation_steps``` indicates that 50 batches of validation images are used per epoch. Workers is set to two, indicating the number of processes used during training. The provided default value is kept, since the network is trained using the Amazon Web Services.

### Training

Using the hyperparameters specified above, the deep neural network is trained. The training curves plotting loss vs epoch for both, training and validation, is shown in the following image:

![alt text][image2]

### Evaluation

Three cases are evaluated,
1. When the quadrotor is following the target.
2. When the quadrotor is in patrol mode and the captured images do not contain the target.
3. When the quadrotor is in patrol mode and the captured images contain the target.

#### Quadrotor is following the target

For this first case:
* The number of true positives, i.e., when the target is correctly identified, is 537.
* The number of false positives, i.e., when an object that is not the target is identified as the target, is 0.
* The number of false negatives, i.e., when the target is not identified as the target, is 2.

Original Image | DNN Prediction
--- | ---
![][image3] | ![][image4]

#### Quadrotor in patrol mode captures images that do not contain the target

For this second case:
* The number of true positives, i.e., when the target is correctly identified, is 0. This is correct since the target does not appear in the images.
* The number of false positives, i.e., when an object that is not the target is identified as the target, is 43. This represents a 16% of error. The total number of non target evaluation images is 270.
* The number of false negatives, i.e., when the target is not identified as the target, is 0. This is correct as well since the target does not appear in the images.

Original Image | DNN Prediction
--- | ---
![][image5] | ![][image6]

#### Quadrotor in patrol mode captures images that contain the target

For this third case:
* The number of true positives, i.e., when the target is correctly identified, is 122.
* The number of false positives, i.e., when an object that is not the target is identified as the target, is 3. This represents a 0.9% of error. The total number of non target evaluation images is 322.
* The number of false negatives, i.e., when the target is not identified as the target, is 179. This represents an error of 55.6%.

An example of a true positive:

Original Image | DNN Prediction
--- | ---
![][image7] | ![][image8]

An example of a false positive:

Original Image | DNN Prediction
--- | ---
![][image9] | ![][image10]

An example of a false negative:

Original Image | Mask | DNN Prediction
--- | --- | ---
![][image11] | ![][image12] | ![][image13] 

#### Results

The evaluation score is obtained from the overall Intersection Over Union -IoU- weighted by true positives. The overall IoU is calculated averaging the IoU of the first and third cases. The weight is calculated by weighting the true positives of all three cases over the true positives, false positives and false negatives of all three cases.

```python
# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print(final_IoU)
```
0.545356017342

```python
# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)
```
0.7437923250564334

```python
# And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```
0.405631620122

### Future Enhancements

From observations to the generated predictions, it can be noted that the majority of errors are produced when either the target is too far, or when the lighting conditions are such that the colors of the target appear darker, or by a combination of both.

When the target is too far, a tiny amount of pixels in the original image are occupied by the target. The neural network then has a hard time associating that little amout of pixels to the target.

An example of how the ligthing conditions darken the target colors is shown as follows:

Original Image | DNN Prediction
--- | --- | ---
![][image14] | ![][image15] 

In this case, the lighting conditions make the lower part of the target appear full almost black, while normally the lower part is a combined pattern of red and black. Hence, the neural network is learning to recognize such pattern, and when that pattern is not perceivable anymore, due to different lighting conditions, the recognition fails. On the other hand, the upper part of the target is recognized, since the lighting conditions did not affect its colors in a significant amount.

To solve this issue, we can 
