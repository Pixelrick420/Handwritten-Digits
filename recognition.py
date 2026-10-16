import numpy as np
import csv
import time
import os
import argparse

IMAGESIZE = 784
HIDDENSIZE1 = 1024
HIDDENSIZE2 = 256
OUTPUTSIZE = 10
ITERATIONS = 40

trainImages = []
trainLabels = []
testImages = []
testLabels = []

"""
First column is the label, remaining are pixel values.
"""
with open('Dataset/mnist_train.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        trainLabels.append(int(row[0]))
        trainImages.append([int(pixel) for pixel in row[1:]])

with open('Dataset/mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        testLabels.append(int(row[0]))
        testImages.append([int(pixel) for pixel in row[1:]])

trainImages = (np.array(trainImages)).T
trainLabels = np.array(trainLabels)
testImages = (np.array(testImages)).T
testLabels = np.array(testLabels)
trainRaw = (trainImages.copy()) / 255.0
testRaw = (testImages.copy()) / 255.0

mean = np.mean(trainImages, axis=1, keepdims=True)
std = np.std(trainImages, axis=1, keepdims=True) + 1e-8
trainImages = (trainImages - mean) / std
testImages = (testImages - mean) / std

"""
trainImages was transposed above, so the shape is (imageSize, nImages).
"""
imageSize, nImages = trainImages.shape

def reLU(vector):
    return np.maximum(0, vector)

def softmax(vector):
    shifted = vector - np.max(vector, axis=0, keepdims=True)
    expVal = np.exp(shifted)
    return expVal / np.sum(expVal, axis=0, keepdims=True)

def derivReLU(vector):
    return (vector > 0)

"""
Weights use He scaling: sqrt(2 / fan_in).
"""
def init():
    W1 = np.random.randn(HIDDENSIZE1, IMAGESIZE) * np.sqrt(2. / IMAGESIZE)
    b1 = np.random.randn(HIDDENSIZE1, 1)

    W2 = np.random.randn(HIDDENSIZE2, HIDDENSIZE1) * np.sqrt(2. / HIDDENSIZE1)
    b2 = np.random.randn(HIDDENSIZE2, 1)

    W3 = np.random.randn(OUTPUTSIZE, HIDDENSIZE2) * np.sqrt(2. / HIDDENSIZE2)
    b3 = np.random.randn(OUTPUTSIZE, 1)

    return W1, b1, W2, b2, W3, b3

def forwardPass(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1.dot(X) + b1 
    A1 = reLU(Z1)

    Z2 = W2.dot(A1) + b2 
    A2 = reLU(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def getExpected(vector): 
    m = vector.size
    expected = np.zeros((OUTPUTSIZE, m))
    for i in range(m):
        expected[vector[i], i] = 1
    return expected

def backPropogate(X, Z1, A1, W1, Z2, A2, W2, Z3, A3, W3, labels):
    N = labels.size
    expected = getExpected(labels)

    dZ3 = A3 - expected
    dW3 = (1 / N) * (dZ3.dot(A2.T))
    db3 = (1 / N) * (np.sum(dZ3, axis=1, keepdims=True))
    
    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * derivReLU(Z2)
    dW2 = (1 / N) * (dZ2.dot(A1.T))
    db2 = (1 / N) * (np.sum(dZ2, axis=1, keepdims=True))

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * derivReLU(Z1)
    dW1 = (1 / N) * (dZ1.dot(X.T))
    db1 = (1 / N) * (np.sum(dZ1, axis=1, keepdims=True))

    return dW1, db1, dW2, db2, dW3, db3

def update(params, grads, velocities, alpha = 0.05, momentum = 0.9):
    newParams = []
    newVelocities = []

    for parameter, gradient, velocity in zip(params, grads, velocities):
        velocity = (momentum * velocity) - (alpha * gradient)
        newParams.append(parameter + velocity)
        newVelocities.append(velocity)

    return newParams, newVelocities

def predict(vector):
    return np.argmax(vector, 0)

def evaluate(predictions, labels):
    return np.sum(predictions == labels) / labels.size

def gradientDescent(images, labels, iterations, alpha = 0.05, batchSize = 256, momentum = 0.9, decay = 1.0, regularization = 0.0):
    params = list(init())
    velocities = [np.zeros_like(parameter) for parameter in params]

    nSamples = images.shape[1]

    bestParams = params
    bestTestAccuracy = 0.0

    for epoch in range(iterations):
        permutation = np.random.permutation(nSamples)
        images = images[:, permutation]
        labels = labels[permutation]

        learningRate = alpha * (decay ** epoch)

        totalBatches = int(np.ceil(nSamples / batchSize))
        reportEvery = max(1, totalBatches // 10)
        epochStart = time.time()

        for batch, j in enumerate(range(0, nSamples, batchSize), start = 1):
            imageBatch = images[:, j:j+batchSize]
            labelBatch = labels[j:j+batchSize]

            Z1, A1, Z2, A2, Z3, A3 = forwardPass(imageBatch, *params)
            grads = list(backPropogate(imageBatch, Z1, A1, params[0], Z2, A2, params[2], Z3, A3, params[4], labelBatch))

            """
            L2 regularization: add weight decay to the gradients.
            """
            if regularization > 0.0:
                for index in (0, 2, 4):
                    grads[index] = grads[index] + (regularization * params[index])

            params, velocities = update(params, grads, velocities, learningRate, momentum)

            if batch % reportEvery == 0 or batch == totalBatches:
                print("Epoch", epoch + 1, "/", iterations, "- batch", batch, "/", totalBatches)

        _, _, _, _, _, trainA3 = forwardPass(trainImages, *params)
        _, _, _, _, _, testA3 = forwardPass(testImages, *params)
        trainAccuracy = evaluate(predict(trainA3), trainLabels)
        testAccuracy = evaluate(predict(testA3), testLabels)
        print(f"Epoch {epoch + 1}/{iterations}  lr {learningRate:.4f}  train {trainAccuracy:.4f}  test {testAccuracy:.4f}  ({time.time() - epochStart:.1f}s)")

        if testAccuracy >= bestTestAccuracy:
            bestTestAccuracy = testAccuracy
            bestParams = [parameter.copy() for parameter in params]

    return (*bestParams, bestTestAccuracy)

def save(W1, b1, W2, b2, W3, b3, mean, std):
    os.makedirs("Parameters", exist_ok=True)

    np.savetxt("Parameters/W1.txt", W1)
    np.savetxt("Parameters/b1.txt", b1)

    np.savetxt("Parameters/W2.txt", W2)
    np.savetxt("Parameters/b2.txt", b2)

    np.savetxt("Parameters/W3.txt", W3)
    np.savetxt("Parameters/b3.txt", b3)

    np.savetxt("Parameters/mean.txt", mean)
    np.savetxt("Parameters/std.txt", std)


def recognize(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forwardPass(X, W1, b1, W2, b2, W3, b3)
    predictions = predict(A3)
    return predictions


def displayImage(image, width=28, height=28):
    chars = " .:-+#@"
    image = image.reshape((height, width))

    for row in image:
        line = ""
        for pixel in row:
            index = int(pixel * (len(chars) - 1))  
            line += chars[index]
        print(line)

def test(index, W1, b1, W2, b2, W3, b3):
    current = trainRaw[:, index, None]
    prediction = recognize(trainImages[:, index, None], W1, b1, W2, b2, W3, b3)
    
    if prediction[0] == trainLabels[index]:
        return True
    else:
        print("Prediction: ", prediction)
        print("Actual : ", trainLabels[index])
        displayImage(current)
        return False

def parseArguments():
    parser = argparse.ArgumentParser(description = "Train the handwritten digit recognition network")
    parser.add_argument("--epochs", type = int, default = ITERATIONS)
    parser.add_argument("--batch-size", type = int, default = 256)
    parser.add_argument("--learning-rate", type = float, default = 0.05)
    parser.add_argument("--momentum", type = float, default = 0.9)
    parser.add_argument("--decay", type = float, default = 0.97)
    parser.add_argument("--regularization", type = float, default = 0.0005)
    return parser.parse_args()

arguments = parseArguments()
W1, b1, W2, b2, W3, b3, accuracy = gradientDescent(
    trainImages, trainLabels, arguments.epochs,
    alpha = arguments.learning_rate, batchSize = arguments.batch_size,
    momentum = arguments.momentum, decay = arguments.decay,
    regularization = arguments.regularization
)
save(W1, b1, W2, b2, W3, b3, mean, std)
np.savez("recognition.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, mean=mean, std=std)

_, _, _, _, _, trainA3 = forwardPass(trainImages, W1, b1, W2, b2, W3, b3)
misclassified = np.where(predict(trainA3) != trainLabels)[0]
print("Misclassified training images :", misclassified.size, "/", nImages)

for index in misclassified[:10]:
    test(index, W1, b1, W2, b2, W3, b3)

final = recognize(testImages, W1, b1, W2, b2, W3, b3)
print("Final test Accuracy :", evaluate(final, testLabels))