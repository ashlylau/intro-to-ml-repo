import numpy as np
import math
import random

# 15526 rows of 6 columns (3 inputs, 3 outputs)
data = np.loadtxt('datasets/robot.dat')
random.shuffle(data)

TOTAL_LEN = len(data) - 10000
TEST_AMT = 500
THRESHOLD = 500  # arbitrary threshold value for calculating accuracy metrics
K = 7

training_set = data[:TOTAL_LEN-TEST_AMT]
test_set = data[-TEST_AMT:]

# calculates eucilidean distance between two rows
def calc_eucilidean_distance(fst, snd):
    dist = 0.0
    for i in range(len(fst)-3):
        dist += (fst[i] - snd[i])**2
    return math.sqrt(dist)

# return map of k nearest neighbours with distances
def get_k_nearest_neighbours(elem):
    distances = list()
    for train_elem in training_set:
        dist = calc_eucilidean_distance(elem, train_elem)
        distances.append((train_elem, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0:K]

# return estimated output values based on mean of neighbours
def estimate_outputs(elem):
    output1 = 0
    output2 = 0
    output3 = 0
    neighbours = get_k_nearest_neighbours(elem)
    for (neighbour, dist) in neighbours:
        output1 += neighbour[3]/K
        output2 += neighbour[4]/K
        output3 += neighbour[5]/K
    return (output1, output2, output3)

# calculate accuracy between predicted and actual output using MSE
def calc_accuracy(pred, actual):
    diff = 0
    for i in range(len(pred)):
        diff += (pred[i] - actual[i])**2
    return math.sqrt(diff)


def demo():
    predictions = list()
    actuals = list()
    for elem in test_set:
        predictions.append(estimate_outputs(elem))
        actuals.append((elem[3], elem[4], elem[5]))
    num_correct = 0
    for i in range(len(predictions)):
        print("pred: " + str(predictions[i]))
        print("actual: " + str(actuals[i]))
        if calc_accuracy(predictions[i], actuals[i]) < THRESHOLD:
            num_correct += 1
    print("num_correct: " + str(num_correct))
    print("accuracy: " + str(num_correct/len(predictions)))
    return num_correct/len(predictions)

demo()
