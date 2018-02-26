# **Traffic Sign Recognition** 

## Writeup

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[n1]: ./new_images/1.jpeg
[n2]: ./new_images/12.jpeg
[n3]: ./new_images/14.jpeg
[n4]: ./new_images/17.jpeg
[n5]: ./new_images/38.jpeg
[barchart]: ./writeup_img/barchart.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/terrylu87/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a bar chart shows how many examples for each class in the training set.

![alt text][barchart]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to put the original images directly into the network. I think turn the image into grayscale will loss color information which might be useful for classification.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Convolution 3x3	    | etc.      									|
| Fully connected		| 4096x120,   outputs 120						|
| RELU					|												|
| Fully connected		| 120x43,   outputs 43							|
| Softmax				|              									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use the Adam optimizer as a default choice, and it works well. Chose batch size of 128, and 40 epochs for training. With the learning rate 0.001, the accuracy increased after almost each epoch in the first 20 epochs. I think tensorflow must support learning rate decay, I'll read more about that later.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.1% 
* test set accuracy of 93.3%

* I start with LeNet-5 as recomended.
* But the accuracy barely reached 90%
* I take alexnet as reference. It uses a lot of 3x3 kernels. I use batch norm after each activation of conv layer. Batch norm have some effect of regulrazation, and makes the modle faster to train.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][n1] ![alt text][n2] ![alt text][n3] 
![alt text][n4] ![alt text][n5]

I think those images are easy to classify, because they are so clean and nice.
But the result confused me.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30km/h      			| 80km/h   										| 
| Priority road 		| Priority road 								|
| Stop					| Priority road									|
| No entry	      		| No entry					 					|
| Keep right			| Keep right      								|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is not good compares  to the accuracy on the test set of 93.3%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 93.61%      			| 80km/h   										| 
| 100.00% 				| Priority road 								|
| 77.34%				| Priority road									|
| 100.00%	      		| No entry					 					|
| 100.00%				| Keep right      								|


For the first an third image, the network fails to classify the right sign. But I think they are clean and nice, later I'll visualizing the output to find some clue.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Have not done yet.
