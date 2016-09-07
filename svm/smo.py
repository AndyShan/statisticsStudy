#coding:utf-8
from numpy import *
import random

def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    data_mat_in = mat(data_mat_in)# x
    label_mat = mat(class_labels).transpose()# y
    b = 0 # 阈值
    m,n = shape(data_mat_in)
    alphas = mat(zeros((m,1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            gx = float(multiply(alphas, label_mat).T * (data_mat_in * data_mat_in[i,:].T)) + b # 对输入的预测值
            ei = gx - float(label_mat[i]) # 预测值和真是输出之差
            if ((label_mat[i] * ei < -toler) and (alphas[i] < c)):# 满足约束条件
                j = select_j_rand(i, m)
                gxj = float(multiply(alphas, label_mat).T * (data_mat_in * data_mat_in[j,:].T)) + b
                ej = gxj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:# 约束条件
                    L = max(0, alphas[j] - alphas[i])
                    H = max(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = max(c, alphas[j] + alphas[i])
                if L == H:
                    print "L == H"
                    continue
                eta = 2.0 * data_mat_in[i,:] * data_mat_in[j,:].T - data_mat_in[i,:].T - data_mat_in[j,:].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alphas[j] -= label_mat[j] * (ei - ej) / eta #  为裁剪最优解
                alphas[j] = clip_alpha(alphas[j], H , L) # 裁剪最优解
                if abs(alphas[j] - alpha_j_old < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat_in[i,:] * data_mat_in[i,:].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * data_mat_in[i,:] * data_mat_in[j,:].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat_in[i,:] * data_mat_in[j,:].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * data_mat_in[j,:] * data_mat_in[j,:].T
                if 0 < alphas[i] and c > alphas[i] :
                    b = b1
                elif 0 < alphas[j] and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha_pairs_changed += 1
                print "iter: %d i: %d,pairs changed %d" % (iter, i, alpha_pairs_changed)
            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print "iteration number: %d" % iter
        return b,alphas


def select_j_rand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

