# -*- coding: utf-8 -*-
from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

# 将一个32*32的数字数据转为1*1024的矩阵
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('C:/Users/AD/statisticsStudy/demo/handwritingDigitTest/digits/trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('C:/Users/AD/statisticsStudy/demo/handwritingDigitTest/digits/trainingDigits/%s' % fileNameStr)
	testFileList = listdir('C:/Users/AD/statisticsStudy/demo/handwritingDigitTest/digits/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('C:/Users/AD/statisticsStudy/demo/handwritingDigitTest/digits/testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		if classifierResult != classNumStr:
			print classifierResult,fileStr
			errorCount += 1.0
	print errorCount/float(mTest)
if __name__ == '__main__':
	handwritingClassTest()