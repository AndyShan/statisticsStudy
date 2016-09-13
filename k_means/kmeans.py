#coding:utf-8
from numpy import *


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat



def dist_eclud(vec_a, vec_b):
    """
    计算两个向量的欧式距离
    :param vec_a:
    :param vec_b:
    :return:
    """
    return sqrt(sum(power(vec_a - vec_b,2)))


def rand_cent(data_set, k):
    """
    构建簇质心
    :param data_set:
    :param k:
    :return:
    """
    n = shape(data_set)[1]  # 获取数据列数
    centroids = mat(zeros((k,n)))  # 构建k列n行的簇质心矩阵
    for j in range(n):
        min_j = min(data_set[:,j])  # 第j列的最小值
        range_j = float(max(data_set[:,j]) - min_j)
        centroids[:,j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids


def k_means(data_set, k, dist_meas = dist_eclud, create_cent = rand_cent):
    m = shape(data_set)[0]  # 输入数据的列数，即待聚类的点的个数
    cluster_assment = mat(zeros((m,2)))  # 聚类结果
    centroids = create_cent(data_set, k)  # 随机初始的簇质心
    cluster_changed = True  # 簇分配结果flag
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j,:],data_set[i,:])   #计算当前点和每一个质心的距离
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            if cluster_assment[i,0] != min_index:  #若改变聚类结果就将flag设为True
                cluster_changed = True
            cluster_assment[i,:] = min_index, min_dist ** 2
        for cent in range(k):
            pts_in_clust = data_set[nonzero(cluster_assment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(pts_in_clust, axis = 0)
    return centroids, cluster_assment

datamat = mat(loadDataSet("testSet1.txt"))
cen, clu = k_means(datamat, 4)
print cen
print clu
