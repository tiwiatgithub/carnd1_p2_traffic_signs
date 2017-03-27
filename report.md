# **Traffic Sign Recognition Report**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/real_examples.png "Real Traffic Sign Examples"
[image5]: ./examples/visualization.png "Visualization of Dataset"
[image6]: ./examples/grayscale.png "Normalized Dataset"
[image7]: ./results/losses.png "Training Losses and Accuracy"
[image8]: ./results/Block_A.jpeg "Inception-ResNet-v1 Block A"
[image9]: ./results/Block_B.jpeg "Inception-ResNet-v1 Block B"
[image10]: ./results/Block_C.jpeg "Inception-ResNet-v1 Block C"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README
Here is a link to my [project code](https://github.com/tiwiatgithub/carnd1_p2_traffic_signs.git)

#### Step 0: Load Image Dataset
The images are provided as pickle data file, so that defined how they should be loaded. The dataset is then  stored for convinience in dictonaries:

```python
for setname in setnames:
    X[setname], y[setname] = data[setname]['features'], data[setname]['labels']
```
The training data can then be accessed as:
```python
X['train']
```
#### Step 1: Print Dataset Information
Next the images are explored. Information about the dataset is printed and some statistics are calculated.
```python
Shape:	             (34799, 32, 32, 3)
Min:	                                0
Max:	                              255
Mean:	                          82.677
Number of training examples:	    34799
Number of testing examples:	     12630
Number of validation examples:	   4410
Number of classes:		             43
```
The classes are represented as number which correspond to certain traffic signs. The encoding is given by 'signnames.csv' and can easily be loaded. I used panda here. The following tables shows the encoding.
```python
ClassId                                           SignName
0         0                               Speed limit (20km/h)
1         1                               Speed limit (30km/h)
2         2                               Speed limit (50km/h)
3         3                               Speed limit (60km/h)
4         4                               Speed limit (70km/h)
5         5                               Speed limit (80km/h)
6         6                        End of speed limit (80km/h)
7         7                              Speed limit (100km/h)
8         8                              Speed limit (120km/h)
9         9                                         No passing
10       10       No passing for vehicles over 3.5 metric tons
11       11              Right-of-way at the next intersection
12       12                                      Priority road
13       13                                              Yield
14       14                                               Stop
15       15                                        No vehicles
16       16           Vehicles over 3.5 metric tons prohibited
17       17                                           No entry
18       18                                    General caution
19       19                        Dangerous curve to the left
20       20                       Dangerous curve to the right
21       21                                       Double curve
22       22                                         Bumpy road
23       23                                      Slippery road
24       24                          Road narrows on the right
25       25                                          Road work
26       26                                    Traffic signals
27       27                                        Pedestrians
28       28                                  Children crossing
29       29                                  Bicycles crossing
30       30                                 Beware of ice/snow
31       31                              Wild animals crossing
32       32                End of all speed and passing limits
33       33                                   Turn right ahead
34       34                                    Turn left ahead
35       35                                         Ahead only
36       36                               Go straight or right
37       37                                Go straight or left
38       38                                         Keep right
39       39                                          Keep left
40       40                               Roundabout mandatory
41       41                                  End of no passing
42       42  End of no passing by vehicles over 3.5 metric ...
```

The get a better idea about the type of data, a random subset is plotted.
![alt text][image5]

I can be observed that all pictures are RGB-colored and some pictures are brighter than others. This already gives a hint, that some preprocessing might be beneficial.

#### Step 2: Preprocessing

As a first step the brightness of all images shall be normalized. I used OpenCV with the algorithm _cv2.equalizeHist_ for that purpose.  
All features (training, testing, validation) are stored within one dictonary, which facilitates the processing by looping over all sets.
Here the function is given as a code example:
```python
def normalize_brightness(features):
    equ = dict()
    for key in features:
        sx, sy = features[key].shape[1], features[key].shape[2]
        equ[key] = np.zeros((features[key].shape[0:3]))
        for idx, img in enumerate(features[key]):
            tmp_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tmp_img = cv2.equalizeHist(tmp_img)
            equ[key][idx,:,:] = tmp_img
        equ[key] = equ[key].reshape(-1, sx, sy, 1)
        print(equ[key].shape)
    return equ
# Normalize all datasets
X = normalize_brightness(X)
```
After the brightness normalization the images are normalized to a [0, 1] range.
Here are some resulting images from the normalization step.
![alt text][image6]

In order to train the model the labels are encoded via _one-hot-encoding_. I used _sklearn.preprocessing.OneHotEncoder_ although there are alternatives in native tensorflow or keras.
I future projects I want to stick to one library and therefore do the entire preprocessing with either tensorflow or keras.

The entire training dataset gets shuffled once, to ensure an even distribution of classes within each batch.

I didn't perform any data augmentation within this project.

#### Step 3: Model Architecture Design

Instead of applying an iterative approach to finding the right architecture, I studied the latest paper of the ILSVRC winning nets and then chose a composed architecture for the projects dataset.
The following paper where the bases of my research:
* _Simonyan, Vedaldi, Zisserman - 2014 - Deep Inside Convolutional Networks Visualising Image Classification Models and Saliency Maps_
* _Springenberg et al. - 2015 - Striving for Simplicity The All Convolutional Net_
* _He et al. - 2015 - Deep Residual Learning for Image Recognition_
* _Windows et al. - 2014 - Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift_
* _Szegedy et al. - 2015 - Going deeper with convolutions_
* _Szegedy et al. - 2016 - Rethinking the Inception Architecture for Computer Vision_
* _Szegedy et al. - 2016 - Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning_

The resulting model architecture is inspire by the work by The Inception-ResNet-v1 model, which yields state-of-the-art results on the ILSVRC challenge and has many interesting design features, which I wanted to explore.

Here is a list of design descision of which I am aware that they are used with Inception-ResNet-v1:
* Batch normalization
* Residual connections
* Inception blocks
* Non-Square convolutional kernels (e.g. [1x7])
* Dimensionality reduction via [1x1] kernels
* Layer concatenation

The sum of these designs leads to a significant reduction of the number of required parameters, which was interesting for the project, where time a computational power are limited.
Here is an code example of an inception block:

```python
def inception_A(x, phase, scope='block_A'):

    nb_filters_1 = 16
    nb_filters_2 = x.get_shape()[3]

    with tf.variable_scope(scope):
        with tf.variable_scope('branch_0'):            
            shortcut = x

        with tf.variable_scope('branch_1'):
            h1 = conv_bn(x, nb_filters_1, [1,1], phase, scope='conv_1_1x1')

        with tf.variable_scope('branch_2'):
            h2a = conv_bn(x,   nb_filters_1, [1,1], phase, scope='conv_2a_1x1')
            h2  = conv_bn(h2a, nb_filters_1, [3,3], phase, scope='conv_2b_3x3')

        with tf.variable_scope('branch_3'):
            h3a = conv_bn(x,   nb_filters_1, [1,1], phase, scope='conv_3a_1x1')
            h3b = conv_bn(h3a, nb_filters_1, [3,3], phase, scope='conv_3b_3x3')
            h3  = conv_bn(h3b, nb_filters_1, [3,3], phase, scope='conv_3c_3x3')
        h = conv_bn((h1+h2+h3), nb_filters_2, [1,1], phase, scope='conv_sum_1x1')
        return tf.nn.relu(shortcut + h, 'relu')
```
The inception blocks used within this project are closed related to the ones given in the original paper.
One major difference is, that I used fewer layers in all of the convolutional layers, because given dataset doesn't require such a large model (or has not enough training examples to train such a large model).

The model used within this project performed quite well, but is likely to be far from optimized.
The model stack looks like this:
```python
h01 = conv_bn(x,   16,  [3, 3], phase, stride=1, scope='layer_h01')
h02 = residuum_stack(h01, 32,  [3, 3], phase, scope='layer_h02')
h04 = inception_A(h02, phase, 'layer_h04')
h05 = conv_bn(h04, 32, [3, 3], phase, stride=2, scope='layer_h05')
h06 = inception_B(h05, phase, 'layer_h06')
h07 = conv_bn(h06, 64, [3, 3], phase, stride=2, scope='layer_h07')
h08 = inception_C(h07, phase, 'layer_h08')
h0e = residuum_stack(h08, 16, [3, 3], phase, scope='layer_h0e')
h00 = tf.contrib.layers.flatten(h0e, scope='flatten')
fc1  = dense_bn(h00, 96, phase, scope='layer_fc1')
fc2  = tf.nn.dropout(fc1, 0.6, name='dropout')
logits = dense(fc2, nb_classes, scope='logits')
```
The model is then train with the entire training set and after each epoch, the accuracy of the testset is evaluated. In case the accuracy increase, a checkpoint is saved.
The parameters for the train processes are:

| Parameter         		|     Value	        					
|:---------------------:|:---------------------------------------------:|
| Epochs         	| 20   |							
| Batch Size     	| 128 |
| Number Batches  | 271 	|
| Learning Rate		| 0.001	|												
| Optimizer	      | Adam 				|

The training is performed on an _AWS E2C g2.2xlarge_ instance and takes approximately 25 minutes to reach a __94.5% accuracy__ on the validation set.
Before each new epoch the batches get shuffled to avoid learning the ordering of images.
Here are the results for each training epoch:

```python
Epochs    Losses  TrainAcc   ValidAcc
0      0.0  0.461417  0.890625  0.467854
1      1.0  0.183732  0.968750  0.546318
2      2.0  0.115782  0.976562  0.744339
3      3.0  0.097919  0.968750  0.860808
4      4.0  0.039325  1.000000  0.883927
5      5.0  0.088541  0.960938  0.883452
6      6.0  0.017706  1.000000  0.864608
7      7.0  0.013156  0.992188  0.790420
8      8.0  0.024263  0.992188  0.932621
9      9.0  0.018978  0.992188  0.904434
10    10.0  0.022867  0.992188  0.891211
11    11.0  0.026390  0.992188  0.910530
12    12.0  0.024211  0.992188  0.908234
13    13.0  0.002738  1.000000  0.892874
14    14.0  0.008251  1.000000  0.932700
15    15.0  0.007410  1.000000  0.941013
16    16.0  0.004334  1.000000  0.889865
17    17.0  0.022935  0.992188  0.903009
18    18.0  0.009998  0.992188  0.937134
19    19.0  0.001920  1.000000  0.949881
```
![alt text][image7]

As can be be seen from the given results, the test accuracy reached 1.0 during the training, which is a sign of overfitting the data.
In order to avoid this and get better validation accuracy, more training data is needed. This can be accomplished to some extend via data augmentation.
However, within this project no augmentation is applied, mainly due to a lack of time.
I spend long hours on adapting and implementing the Inception-ResNet-v1 architecture and had some AWS NVIDIA driver issue close to the deadline, which contributed to the delay.

### Test a Model on New Images
Here are some images for testing the model.
![alt text][image4]
