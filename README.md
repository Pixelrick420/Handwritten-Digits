# Handwritten Digit Recognition using a Fully Connected Neural Network

## Overview

This project implements a handwritten digit recognition system trained on the MNIST dataset using a custom neural network built from scratch with NumPy. A GUI built with Tkinter allows users to draw digits for real-time prediction using the trained model.

I learned all of the maths for implementing this from [3blue1brown](https://www.3blue1brown.com/topics/neural-networks)

## Dataset

The [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) consists of 60,000 training images and 10,000 testing images. Each image is a grayscale 28x28 pixel representation of a handwritten digit (0–9). The dataset is used in CSV format where the first column represents the digit label and the remaining 784 columns represent pixel intensities.

The CSV files are not committed to this repository. Download the official IDX files into a `Dataset/` folder and convert them:

```bash
mkdir -p Dataset
cd Dataset
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
cd ..
```

Then convert them to the CSV layout `recognition.py` expects — **label in the first column**, followed by the 784 pixel intensities, with a header row:

```bash
python3 - <<'PY'
import gzip, struct
import numpy as np

def readIdx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        dimensions = magic % 256
        sizes = struct.unpack(">" + "I" * dimensions, f.read(4 * dimensions))
        return np.frombuffer(f.read(), dtype = np.uint8).reshape(sizes)

def writeCsv(images, labels, out):
    pixels = images.reshape(len(labels), -1)
    data = np.column_stack([labels, pixels])
    header = "label," + ",".join(f"pixel{i}" for i in range(pixels.shape[1]))
    np.savetxt(out, data, fmt = "%d", delimiter = ",", header = header, comments = "")

writeCsv(readIdx("Dataset/train-images-idx3-ubyte.gz"), readIdx("Dataset/train-labels-idx1-ubyte.gz"), "Dataset/mnist_train.csv")
writeCsv(readIdx("Dataset/t10k-images-idx3-ubyte.gz"), readIdx("Dataset/t10k-labels-idx1-ubyte.gz"), "Dataset/mnist_test.csv")
print("wrote Dataset/mnist_train.csv and Dataset/mnist_test.csv")
PY
```

This produces `Dataset/mnist_train.csv` (60,000 rows) and `Dataset/mnist_test.csv` (10,000 rows), which is exactly where `recognition.py` looks for them.

## Architecture

The neural network follows a feedforward architecture with the following layers:

- Input Layer: 784 neurons (corresponding to the 28×28 image size)
- First Hidden Layer: 1024 neurons with ReLU activation
- Second Hidden Layer: 256 neurons with ReLU activation
- Output Layer: 10 neurons with Softmax activation

The model is trained using mini-batch gradient descent with a batch size of 256, momentum, a decaying learning rate, and L2 regularization. The parameters scoring the best test accuracy are kept, and train/test accuracy is printed after every epoch.

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
- `requirements.txt`: Python dependencies
- `Dataset/`: MNIST CSV files (not committed)

## Accuracy

- Final test accuracy: **~98.1%**
- Training accuracy is evaluated at regular intervals during training
- The best-by-test parameters are saved, so the exported model reflects the best epoch rather than the last one

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

1. (Optional) Create a virtual environment and install the dependencies:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

> Note: the GUI requires `scipy` and `tkinter`. Some systems ship a Python without `scipy`, so installing from `requirements.txt` (in a venv) is recommended.

2. Train the model by running:

```bash
venv/bin/python recognition.py
```

This will generate a `recognition.npz` file containing all model parameters (and the `W1.txt`…`b3.txt`, `mean.txt`, `std.txt` text dumps in `Parameters/`).

Training hyperparameters can be overridden from the command line:

```bash
venv/bin/python recognition.py --epochs 40 --batch-size 256 --learning-rate 0.05 \
    --momentum 0.9 --decay 0.97 --regularization 0.0005
```

| Flag | Default | Description |
| --- | --- | --- |
| `--epochs` | `40` | Number of passes over the training set |
| `--batch-size` | `256` | Images per mini-batch |
| `--learning-rate` | `0.05` | Initial learning rate |
| `--momentum` | `0.9` | Momentum coefficient |
| `--decay` | `0.97` | Per-epoch learning-rate decay |
| `--regularization` | `0.0005` | L2 weight-decay strength |

3. Launch the GUI by running:

```bash
venv/bin/python app.py
```

Ensure the model file (`recognition.npz`) is in the same directory.

## Future Work

- Improve accuracy with convolutional layers
- Allow user to load their own images
- Experiment with different layer sizes and architectures

## License

This project is open-source and free to use
