# -*- coding: utf-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0] # 获取训练样本集的一组数据的长度
	diffMat = tile(inX,(dataSetSize,1)) - dataSet # 将输入数据减去训练样本集
	sqDiffMat = diffMat ** 2 # 计算每一个差的平方
	sqDistances = sqDiffMat.sum(axis = 1) # 将矩阵每一行的元素求和，即求差的平方的和
	distances = sqDistances ** 0.5 # 开方
	sortedDistIndicies = distances.argsort() # 根据distances的值进行升序排序,返回排序后的元素下标
	classCount = {}
	for i in range(k): # 在k的范围内进行循环
		voteIlabel = labels[sortedDistIndicies[i]] # 按下标索引返回邻域中第i个近邻点的分类
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 统计分类的个数(字典实现)
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True) #以分类的个数为关键字降序排列
	return sortedClassCount[0][0]

"""将文件读入"""
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         
    returnMat = zeros((numberOfLines,3))        
    classLabelVector = []                       
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':#用1,2,3代表三个分类
        	a = 1
        elif listFromLine[-1] == 'smallDoses':
        	a = 2
        elif listFromLine[-1] == 'largeDoses':
        	a = 3
        classLabelVector.append(a)
        index += 1
    return returnMat,classLabelVector

"""特征值相除进行数值归一化
公式：newValue = (oldValue - min) / (max - min)"""
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

def visualization(datingDataMat,datingLabels):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
	plt.show()	

def datingClassTest(filename):
	hoRatio = 0.10
	datingDataMat,datingLabels = file2matrix(filename)
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print errorCount / float(numTestVecs)

if __name__ == "__main__":
	datingDataMat,datingLabels = file2matrix('C:/Users/AD/statisticsStudy/demo/datingTest/datingTestSet.txt')
	visualization(datingDataMat,datingLabels)
	datingClassTest('C:/Users/AD/statisticsStudy/demo/datingTest/datingTestSet.txt')