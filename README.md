# PRODIGY_GA_05
# Neural Style Transfer

This project applies the artistic style of one image (e.g., a famous painting) to the content of another image using Neural Style Transfer techniques.

## Requirements

Make sure you have the following libraries installed:

- TensorFlow
- Keras
- NumPy

You can install these libraries using pip:

```bash
pip install tensorflow keras numpy
```
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19

# Load base and style images
base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimensions of the generated picture
width, height = keras.utils.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
```
## How It Works

1. **Load Images**: The base image and style reference image are downloaded from provided URLs.

2. **Set Weights**: You can adjust the weights for total variation, style, and content to control the effect of each in the final image.

3. **Define Dimensions**: The dimensions for the generated image are set based on the base image.


## Displaying Images

You can visualize the base image and style reference image using the following code:

```python
from IPython.display import Image, display

# Display the base image
display(Image(base_image_path))

# Display the style reference image
display(Image(style_reference_image_path))
```
## Image Preprocessing / Deprocessing Utilities

To prepare the images for the Neural Style Transfer process, you'll need to preprocess them and later deprocess the generated output. Here are the utility functions for these tasks:

``` python
def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.utils.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
```
## Compute the Style Transfer Loss

To perform style transfer, we need to define four utility functions:

1. **Gram Matrix**: Used to compute the style loss.
2. **Style Loss**: Keeps the generated image close to the local textures of the style reference image.
3. **Content Loss**: Ensures that the high-level representation of the generated image is similar to that of the base image.
4. **Total Variation Loss**: A regularization loss that keeps the generated image locally coherent.

### Utility Functions

```python
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
```
## Feature Extraction Model

Next, let's create a feature extraction model that retrieves the intermediate activations of VGG19 as a dictionary by name.

### Code

```python
# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```
## Compute Style Transfer Loss

Finally, here's the code that computes the style transfer loss.

### Code

```python
# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

# The layer to use for the content loss.
content_layer_name = "block5_conv2"

def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = 
```
## Add a TensorFlow Function Decorator

To optimize the loss and gradient computation, we can compile the function using the `@tf.function` decorator, which makes it faster.

### Code

```python
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads
```
## The Training Loop

In this section, we repeatedly run vanilla gradient descent steps to minimize the loss and save the resulting image every 100 iterations. We also decay the learning rate by 0.96 every 100 steps.

### Code

```python
# Set up the optimizer with exponential decay
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

# Preprocess the images
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

# Training iterations
iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.utils.save_img(fname, img)
```
## Result After Training

After 4000 iterations, you will obtain the following result:

### Code

```python
display(Image(result_prefix + "_at_iteration_4000.png"))
```

This code displays the final stylized image after completing the training loop for 4000 iterations. You should see the combination of the content from the base image and the style from the style reference image.







 
