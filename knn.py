import os
import numpy as np
from collections import Counter
from utils import load_mnist

path = os.path.join(os.getcwd(), 'dataset')
train_x, train_y = load_mnist(path, 'train')
test_x, test_y = load_mnist(path, 't10k')

print('train images: ', train_x.shape[0])
print('test images: ', test_x.shape[0])


def classify(input, k, train_x, train_y):
    input = input / 255.0  # normalize
    train_x = train_x / 255.0  # normalize
    dists = []
    for i in range(train_x.shape[0]):
        dist = np.linalg.norm(train_x[i] - input)
        dists.append(dist)
    dists = np.array(dists)
    sored_idx = np.argsort(dists)
    class_list = []
    for idx in sored_idx[:k]:
        class_list.append(train_y[idx])
    counter = Counter(class_list)
    most_common = counter.most_common(1)
    for label, num in most_common:
        return label, num  # return most common label, which is the result of KNN


def knn(train_x, train_y, test_x, test_y, k):
    correct_num = 0
    for i in range(test_x.shape[0]):
        label, num = classify(test_x[i], k, train_x, train_y)
        if label == test_y[i]:
            correct_num += 1
        acc = correct_num/(i+1)
        print('k:', k, 'test case:', i, 'predict:', label, 'ground truth:', test_y[i], 'acc:', acc)
    return correct_num / test_x.shape[0]


if __name__ == "__main__":
    for k in range(3, 9):
        acc = knn(train_x, train_y, test_x, test_y, k)
        print('KNN k:', k, 'acc:', acc)
        with open('knn_result.txt', 'a', encoding='utf-8') as f:
            f.write('KNN k: ' + str(k) + ' acc: ' + str(acc) + '\n')
