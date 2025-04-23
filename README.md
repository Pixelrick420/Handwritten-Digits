
# Handwritten Digit Recognition using a Fully Connected Neural Network

## Overview

This project implements a handwritten digit recognition system trained on the MNIST dataset using a custom neural network built from scratch with NumPy. A GUI built with Tkinter allows users to draw digits for real-time prediction using the trained model.

I learned all of the maths for implementing this from [3blue1brown](https://www.3blue1brown.com/topics/neural-networks)

## Dataset

The [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) consists of 60,000 training images and 10,000 testing images. Each image is a grayscale 28x28 pixel representation of a handwritten digit (0–9). The dataset is used in CSV format where the first column represents the digit label and the remaining 784 columns represent pixel intensities.

## Architecture

The neural network follows a basic feedforward architecture with the following layers:

- Input Layer: 784 neurons (corresponding to the 28×28 image size)
- Hidden Layer: 1024 neurons with ReLU activation
- Output Layer: 10 neurons with Softmax activation

The model is trained using mini-batch gradient descent.

### Mathematics

#### ReLU Activation
The ReLU activation function is defined as:

$$
	ext{ReLU}(x) = \max(0, x)
$$

#### Softmax Activation
The softmax function outputs a probability distribution:

$$
	ext{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

#### Forward Pass
For an input \( X \), the network computes:

$$
Z_1 = W_1X + b_1 \\
A_1 = 	ext{ReLU}(Z_1) \\
Z_2 = W_2A_1 + b_2 \\
A_2 = 	ext{Softmax}(Z_2)
$$

#### Loss Function
Cross-entropy loss is minimized using gradient descent with backpropagation.

#### Backpropagation
The gradients for each layer are computed using:

$$
dZ_2 = A_2 - Y \\
dW_2 = \frac{1}{m}dZ_2 A_1^T \\
db_2 = \frac{1}{m} \sum dZ_2 \\
dZ_1 = W_2^T dZ_2 * 	ext{ReLU}'(Z_1) \\
dW_1 = \frac{1}{m}dZ_1 X^T \\
db_1 = \frac{1}{m} \sum dZ_1
$$

## GUI Application

The GUI allows the user to draw a digit, which is then:

- Centered using the center of mass
- Denoised by removing low intensity noise
- Normalized using training set statistics
- Passed through the neural network for classification

### Features

- Real-time digit drawing on a canvas
- Display of predicted digit
- Uses saved weights and biases

### Libraries Used

- `numpy`
- `tkinter`
- `scipy.ndimage`

## File Structure

- `W1.txt`, `W2.txt`, `b1.txt`, `b2.txt`: Model weights and biases
- `mean.txt`, `std.txt`: Normalization parameters
- `app.py`: Tkinter-based GUI for drawing and recognizing digits
- `recognition.py`: Neural network training and saving

## Accuracy

- Final test accuracy: **~97%**
- Training accuracy is evaluated at regular intervals during training

## Example GUI & Output

### GUI



### Output
- Training log

- Drawing '5' on the canvas yields prediction: **5**

- Drawing '3' yields prediction: **3**

## Usage

1. Train the model by running:

```bash
python recognition.py
```
You can also use the default weights. Or try tweaking the code to get better performance.

2. Launch the GUI by running:

```bash
python app.py
```

Ensure model weights and parameters (`W1.txt`, `b1.txt`, etc.) are in the same directory.

## Future Work

- Improve accuracy with convolutional layers
- Save training progress with checkpoints
- Allow user to load their own images

## License

This project is open-source and free to use