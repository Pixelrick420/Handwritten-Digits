# Handwritten Digit Recognition using a Fully Connected Neural Network

## Overview

This project implements a handwritten digit recognition system trained on the MNIST dataset using a custom neural network built from scratch with NumPy. A GUI built with Tkinter allows users to draw digits for real-time prediction using the trained model.

I learned all of the maths for implementing this from [3blue1brown](https://www.3blue1brown.com/topics/neural-networks)

## Dataset

The [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) consists of 60,000 training images and 10,000 testing images. Each image is a grayscale 28x28 pixel representation of a handwritten digit (0–9). The dataset is used in CSV format where the first column represents the digit label and the remaining 784 columns represent pixel intensities.

## Architecture

The neural network follows a feedforward architecture with the following layers:

- Input Layer: 784 neurons (corresponding to the 28×28 image size)
- First Hidden Layer: 512 neurons with ReLU activation
- Second Hidden Layer: 192 neurons with ReLU activation
- Output Layer: 10 neurons with Softmax activation

The model is trained using mini-batch gradient descent with a batch size of 64.

### Mathematics

## ReLU Activation
The ReLU (Rectified Linear Unit) activation function is defined as:

```math
\text{ReLU}(x) = \max(0, x)
```

## Softmax Activation
The softmax function outputs a probability distribution over classes:

```math
\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
```

## Forward Pass
For an input vector \( X \), the forward propagation steps are:

```math
Z^{[1]} = W^{[1]}X + b^{[1]}
```

```math
A^{[1]} = \text{ReLU}(Z^{[1]})
```

```math
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
```

```math
A^{[2]} = \text{ReLU}(Z^{[2]})
```

```math
Z^{[3]} = W^{[3]}A^{[2]} + b^{[3]}
```

```math
A^{[3]} = \text{Softmax}(Z^{[3]})
```

## Loss Function
The loss function used is cross-entropy loss:

```math
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} Y^{(i)} \log(A^{[3](i)})
```

## Backpropagation
The gradients are computed as follows:

```math
dZ^{[3]} = A^{[3]} - Y
```

```math
dW^{[3]} = \frac{1}{m} dZ^{[3]} A^{[2]^T}
```

```math
db^{[3]} = \frac{1}{m} \sum dZ^{[3]}
```

```math
dA^{[2]} = W^{[3]^T} dZ^{[3]}
```

```math
dZ^{[2]} = dA^{[2]} * \text{ReLU}'(Z^{[2]})
```

```math
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]^T}
```

```math
db^{[2]} = \frac{1}{m} \sum dZ^{[2]}
```

```math
dA^{[1]} = W^{[2]^T} dZ^{[2]}
```

```math
dZ^{[1]} = dA^{[1]} * \text{ReLU}'(Z^{[1]})
```

```math
dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T
```

```math
db^{[1]} = \frac{1}{m} \sum dZ^{[1]}
```

## GUI Application

The GUI allows the user to draw a digit, which is then:

- Centered using the center of mass
- Denoised by removing low intensity noise
- Normalized using training set statistics
- Passed through the neural network for classification

### Features

- Real-time digit drawing on a canvas
- Display of predicted digit
- Uses saved model parameters from an .npz file

### Libraries Used

- `numpy`
- `tkinter`
- `scipy.ndimage`

## File Structure

- `recognition.npz`: Contains all model weights, biases, and normalization parameters
- `app.py`: Tkinter-based GUI for drawing and recognizing digits
- `recognition.py`: Neural network training and saving

## Accuracy

- Final test accuracy: **~98%**
- Training accuracy is evaluated at regular intervals during training

## Example GUI & Output

### GUI

![image](https://github.com/user-attachments/assets/27617268-820e-46c2-a8af-1c37e58dce1b)

### Output
- Training log

![image](https://github.com/user-attachments/assets/527d3cb3-ae49-486f-9465-c7812d6ce2dc)


- Drawing '5' on the canvas yields prediction: **5**
  
![image](https://github.com/user-attachments/assets/ddb5e584-d6b8-45d4-9368-6934426ce87e)

- Drawing '3' yields prediction: **3**

![image](https://github.com/user-attachments/assets/21e5dfb9-9b12-47d8-9524-23257c1d8af4)

## Usage

1. Train the model by running:

```bash
python recognition.py
```
This will generate a `recognition.npz` file containing all model parameters.

2. Launch the GUI by running:

```bash
python app.py
```

Ensure the model file (`recognition.npz`) is in the same directory.

## Future Work

- Improve accuracy with convolutional layers
- Save training progress with checkpoints
- Allow user to load their own images
- Experiment with different layer sizes and architectures

## License

This project is open-source and free to use
