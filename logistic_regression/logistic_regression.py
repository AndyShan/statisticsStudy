#coding:utf-8
from math import exp

from numpy import *


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(input_x):
    return 1.0/(1 + exp(-input_x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m,n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n,1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stoc_grad_ascent(data_matrix, class_labels):
    m,n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights += alpha * error * data_matrix[i]
    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=500):
    m,n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01#随着迭代减小alpha，减小alpha的影响，有常数保证永不为0，保证新数据仍然具有影响
            rand_index = int(random.uniform(0,len(data_index)))#随机选取样本点更新数据，避免周期性波动
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights
