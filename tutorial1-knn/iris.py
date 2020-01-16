import numpy as np
import math
import random

# 150 rows of 7 columns (4 attributes, 3 cols of one-hot encoding for species)
data = np.loadtxt('datasets/iris.dat')
random.shuffle(data)

training_set = data[0:120]
test_set = data[-30:]

K = 9

# calculates eucilidean distance between two rows
def calc_eucilidean_distance(fst, snd):
    dist = 0.0
    for i in range(len(fst)-3):
        dist += (fst[i] - snd[i])**2
    return math.sqrt(dist)

# return map of k nearest neighbours with distances and species category
def get_k_nearest_neighbours(elem):
    distances = list()
    for train_elem in training_set:
        dist = calc_eucilidean_distance(elem, train_elem)
        distances.append((train_elem, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0:K]

# get species (1, 2 or 3) from row
def get_species(elem):
    if round(elem[4]) == 1:
        return 1
    if round(elem[5]) == 1:
        return 2
    if round(elem[6]) == 1:
        return 3

# return estimated species based on species (and weights) of neighbours
def estimate_species(elem, weight):
    neighbours = get_k_nearest_neighbours(elem)
    species_map = dict({1:0, 2:0, 3:0})
    for (neighbour, dist) in neighbours:
        if weight:
            species_map[get_species(neighbour)] += get_weight(dist)
        else:
            species_map[get_species(neighbour)] += 1
    return max(species_map, key=species_map.get)

# calculate weight = inverse of distance
def get_weight(dist):
    return 1/(dist+0.00001)

def demo(weight=False):
    predictions = list()
    actuals = list()
    for elem in test_set:
        predictions.append(estimate_species(elem, weight))
        actuals.append(get_species(elem))
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == actuals[i]:
            num_correct += 1
    print("predictions: " + str(predictions))
    print("actuals: " + str(actuals))
    print("num_correct: " + str(num_correct))
    print("accuracy: " + str(num_correct/len(predictions)))
    return num_correct/len(predictions)


print("====NORMAL====")
demo()
print("====WEIGHTS====")
demo(True)
