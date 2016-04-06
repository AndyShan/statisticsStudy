# -*- coding: utf-8 -*-
from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group , labels

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] #获取训练样本集的一组数据的长度
	diffMat = tile(inX,(dataSetSize,1)) - dataSet #将输入数据减去训练样本集
	sqDiffMat = diffMat ** 2 #计算每一个差的平方
	sqDistances = sqDiffMat.sum(axis = 1) #将矩阵每一行的元素求和，即求差的平方的和
	distances = sqDistances ** 0.5 #开方
	sortedDistIndicies = distances.argsort() #根据distances的值进行升序排序,返回排序后的元素下标
	classCount = {}
	for i in range(k): # 在k的范围内进行循环
		voteIlabel = labels[sortedDistIndicies[i]] # 按下标索引返回邻域中第i个近邻点的分类
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 统计分类的个数(字典实现)
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True) #以分类的个数为关键字降序排列
	return sortedClassCount[0][0]

if __name__ == "__main__":
	group,labels = createDataSet()
	a = classify0([0,0],group,labels,3)
	print a



