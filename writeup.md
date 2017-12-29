## Project: Deep Learning Follow me

---


**Steps to complete the project:**  


1. Clone the [project repo](https://github.com/udacity/RoboND-DeepLearning-Project.git).
2. Fill out the TODO's in the project code.
3. Optimize your network and hyper-parameters.
4. Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
5. Make a brief writeup report summarizing why you made the choices you did in building the network.


[//]: # (Image References)

[network_diagram]: ./writeup_images/Network.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

#### 2. The write-up conveys the an understanding of the network architecture.
The network designed for this project is made of 3 parts:
1- Encoder
2- 1x1 convolution layer
3- Decoder

The Encoder consists of separable convolution layers, after experimenting 4 layers and 64 filters for each gave me the best results:
```
encoded_layer_1 = encoder_block(inputs, 64, 2)
encoded_layer_2 = encoder_block(encoded_layer_1, 64, 2)
encoded_layer_3 = encoder_block(encoded_layer_2, 64, 2)
encoded_layer_4 = encoder_block(encoded_layer_3, 64, 2)
```
After that we have a layer of 1x1 convolutional layer:
```
conv2d_layer = conv2d_batchnorm(encoded_layer_4, 64, kernel_size=1, strides=1)
```
Then a decoder is used which contains 4 layers same as the encoder, in each decoding step, an encoded layer is upsampled and then concatenated to another layer which has more information than the upsampled one to obtain better results, after that the concatenation output goes  through two separable convolution layers.
```
decoded_layer_1 = decoder_block(conv2d_layer, encoded_layer_3, 64)
decoded_layer_2 = decoder_block(decoded_layer_1, encoded_layer_2, 64)
decoded_layer_3 = decoder_block(decoded_layer_2, encoded_layer_1, 64)
decoded_layer_4 = decoder_block(decoded_layer_3, inputs, 64)
```
```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
  # Upsample the small input layer using the bilinear_upsample() function.
  upsampled_small_layer = bilinear_upsample(small_ip_layer)
  
  # Concatenate the upsampled and large input layers using layers.concatenate
  output = layers.concatenate([upsampled_small_layer, large_ip_layer])
  
  # Add some number of separable convolution layers
  output_layer_1 = separable_conv2d_batchnorm(output, filters)
  output_layer = separable_conv2d_batchnorm(output_layer_1, filters)
  return output_layer
```
![network diagram][network_diagram]
#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.
The hyperparameters had to be tuned to obtain an acceptable score, first of all the learning rate had to be reduced gradually and the number of epochs increased gradually. Although these changes resulted in longer training time but it increased the network's score until a score greater than 0.4 was obtained.
```
learning_rate = 0.001
batch_size = 100
num_epochs = 12
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.
In FCN instead of feading the output of the convolution network into a fully connected layer, a 1X1 convolution layer is used instead, this way we won't have to flatten the output into a 2D tensor and accordingly we won't be losing spatial information.
Also the replacement of fully-connected layers with convolutional layers presents an added advantage that during inference, we can feed images of any size into our trained network.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.
As mentioned above in the network architecture section, the encoder is made of 4 convolutions, each convolution layer increases the level of details the network is capturing and improves its ability to identify the understand the input image and accordingly identify the hero.
Then comes the decoder which upsamples the shrinked input passed to it by the encoder, the decoder uses layers concatenation to apply skip connection techniques, which retains some of the data lost while encoding.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.
As data collection is as important as building the model, the current model and data won't be able to follow a different object, like a dog or a cat for example, because the data (images) was collected to help the model identify the human hero, it captured images with the hero in it from different angles as well as images in which the hero walks among other humans and so on. If we want to follow a dog for example, we'll have to collect new data, capturing the dog from different angles and in different positions as well. And then train our model using the new data.

### Model

#### 1. The model is submitted in the correct format.
The model is saved in `/data/weights/model_weights`

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.
The network achieved a score of 0.4263