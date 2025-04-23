import numpy as np
import csv
import random

"""
Constants for neural network layer sizes
"""
IMAGESIZE = 784
HIDDENSIZE = 1024
OUTPUTSIZE = 10

"""
Global variables for datasets
"""
trainImages = []
trainLabels = []
testImages = []
testLabels = []

"""
Load training data from CSV
First column is the label, remaining are pixel values
"""
with open('D://Code//Recognition//Dataset//mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        trainLabels.append(int(row[0]))
        trainImages.append([int(pixel) for pixel in row[1:]])

"""
Load testing data from CSV
"""
with open('D://Code//Recognition//Dataset//mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        testLabels.append(int(row[0]))
        testImages.append([int(pixel) for pixel in row[1:]])

"""
Convert lists to numpy arrays
"""
trainImages = (np.array(trainImages)).T
trainLabels = np.array(trainLabels)
testImages = (np.array(testImages)).T
testLabels = np.array(testLabels)
trainRaw = (trainImages.copy()) / 255.0
testRaw = (testImages.copy()) / 255.0

"""
Pre process data and normalize pixel values
""" 
mean = np.mean(trainImages, axis=1, keepdims=True)
std = np.std(trainImages, axis=1, keepdims=True) + 1e-8
trainImages = (trainImages - mean) / std
testImages = (testImages - mean) / std

"""
Get dataset shape: nImages = 60000, imageSize = 784
"""
nImages, imageSize = trainImages.shape

"""
ReLU activation function
"""
def reLU(vector):
    return np.maximum(0, vector)

"""
Softmax activation function for output layer
"""
def softmax(vector):
    shifted = vector - np.max(vector, axis=0, keepdims=True)
    expVal = np.exp(shifted)
    return expVal / np.sum(expVal, axis=0, keepdims=True)

"""
Derivative of ReLU function
"""
def derivReLU(vector):
    return (vector > 0)

"""
Initialize weights and biases with random values
W1: weights from input to hidden layer
b1: biases for hidden layer
W2: weights from hidden to output layer
b2: biases for output layer
"""
def init():
    W1 = np.random.randn(HIDDENSIZE, IMAGESIZE) * np.sqrt(2. / IMAGESIZE)
    W2 = np.random.randn(OUTPUTSIZE, HIDDENSIZE) * np.sqrt(2. / HIDDENSIZE)
    b1 = np.random.randn(HIDDENSIZE, 1)
    b2 = np.random.randn(OUTPUTSIZE, 1)
    return W1, b1, W2, b2

"""
Forward pass through the network
Returns intermediate outputs for backpropagation
"""
def forwardPass(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 
    A1 = reLU(Z1)
    Z2 = W2.dot(A1) + b2 
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

"""
Convert numeric labels to one-hot encoded format
"""
def getExpected(vector): 
    m = vector.size
    expected = np.zeros((OUTPUTSIZE, m))
    for i in range(m):
        expected[vector[i], i] = 1
    return expected

"""
Perform backpropagation to compute gradients
Returns gradients for all weights and biases
"""
def backPropogate(Z1, A1, Z2, A2, W2, X, labels):
    expected = getExpected(labels)
    dZ2 = A2 - expected
    dW2 = (1 / labels.size) * (dZ2.dot(A1.T))
    db2 = (1 / labels.size) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    dW1 = (1 / labels.size) * (dZ1.dot(X.T))
    db1 = (1 / labels.size) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

"""
Update weights and biases using computed gradients
"""
def update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha = 0.1):
    W1  = W1 - (alpha * dW1)
    W2 = W2 - (alpha * dW2)
    b1 = b1 - (alpha * db1)
    b2 = b2 - (alpha * db2)
    return W1, b1, W2, b2

"""
Return the predicted class index with the highest probability
"""
def predict(vector):
    return np.argmax(vector, 0)

"""
Evaluate accuracy by comparing predictions to ground-truth labels
"""
def evaluate(predictions, labels):
    return np.sum(predictions == labels) / labels.size

"""
Train the model using gradient descent
Performs shuffling of data after each iteration
Prints accuracy every 50 iterations
"""
def gradientDescent(images, labels, iterations, alpha = 0.1, batchSize = 64):
    W1, b1, W2, b2 = init()
    for _ in range(iterations):
        """
        Shuffle data after each full pass
        """
        permutation = np.random.permutation(images.shape[1])
        images = images[:, permutation]
        labels = labels[permutation]

        """
        Process in batches
        """
        for j in range(0, nImages, batchSize):
            imageBatch = images[:, j:j+batchSize]
            labelBatch = labels[j:j+batchSize]

            Z1, A1, Z2, A2 = forwardPass(W1, b1, W2, b2, imageBatch)
            dW1, db1, dW2, db2 = backPropogate(Z1, A1, Z2, A2, W2, imageBatch, labelBatch)
            W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        """
        Display progress every 50 iterations
        """
        if (_ % 50 == 0):
            print("Iteration :", _)
            _, _, _, testA2 = forwardPass(W1, b1, W2, b2, trainImages)
            print("Accuracy :", evaluate(predict(testA2), trainLabels))
    
        elif (_ >= iterations - 1):
            print("Iteration :", iterations)
            _, _, _, testA2 = forwardPass(W1, b1, W2, b2, trainImages)
            acc = evaluate(predict(testA2), trainLabels)
            print("Final Accuracy :", acc)

    return W1, b1, W2, b2, acc

"""
Save weights, biases, and normalization parameters to text files
"""
def save(W1, b1, W2, b2, mean, std):
    np.savetxt("W1.txt", W1)
    np.savetxt("W2.txt", W2)
    np.savetxt("b1.txt", b1)
    np.savetxt("b2.txt", b2)
    np.savetxt("mean.txt", mean)
    np.savetxt("std.txt", std)


"""
Recognize an image and predict the digit 
"""
def recognize(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardPass(W1, b1, W2, b2, X)
    predictions = predict(A2)
    return predictions


"""
Display an image using ascii art
"""
def displayImage(image, width=28, height=28):
    chars = " .:-+#@"
    image = image.reshape((height, width))

    for row in image:
        line = ""
        for pixel in row:
            index = int(pixel * (len(chars) - 1))  
            line += chars[index]
        print(line)

"""
Test the model using an image at a given index
"""
def test(index, W1, b1, W2, b2):
    current = trainRaw[:, index, None]
    prediction = recognize(trainImages[:, index, None], W1, b1, W2, b2)
    print("Prediction: ", prediction)
    print("Actual : ", trainLabels[index])
    displayImage(current)

"""
Train the network and save the trained model if its accuracy is good, else do it again
"""
W1, b1, W2, b2, accuracy = gradientDescent(trainImages, trainLabels, 500)
save(W1, b1, W2, b2, mean, std)

for _ in range(10):
    test(random.randint(0, nImages), W1, b1, W2, b2)

"""
Test accuracy against test data
"""
final = recognize(testImages, W1, b1, W2, b2)
print("Final final Accuracy :", evaluate(final, testLabels))