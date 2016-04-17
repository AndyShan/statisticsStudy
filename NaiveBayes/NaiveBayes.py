# -*- coding: utf-8 -*-
from numpy import *

"""初始化数据"""
def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],\
				  ['maybe','not','take','him','to','dog','park','stupid'],\
				  ['my','dalmation','is','so','cute','I','love','him'],\
				  ['stop','posting','stupid','worthless','garbage'],\
				  ['mr','licks','ate','my','steak','how','to','stop','him'],\
				  ['quit','buying','worthless','dog','food','stupid']]# 词条
	classVec = [0,1,0,1,0,1]# 词条的分类
	return postingList,classVec

"""创建不重复的单词库"""
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

"""将单词转为向量(词集模型)"""
def setOfWord2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1# 若有则为1
		# else:
		# 	print "the word: %s is not in my Vocabulary!"% word
	return returnVec

"""将单词转为向量(词袋模型)"""
def bagofWord2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1#若有则+1
	return returnVec

"""朴素贝叶斯分类器训练函数"""
def trainNB0(trainMarix,trainCategory):
	numTrainDocs = len(trainMarix)
	numWords = len(trainMarix[0]) 
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMarix[i]
			p1Denom += sum(trainMarix[i])

		else:
			p0Num += trainMarix[i]
			p0Denom += sum(trainMarix[i])

	p1Vect = log(p1Num / p1Denom)
	p0Vect = log(p0Num / p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

if __name__ == '__main__':
	listOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
	print trainMat
	p0v,p1v,pAb = trainNB0(trainMat,listClasses)
	testEntry = ['love','my','dalmation']
	thisDoc = array(setOfWord2Vec(myVocabList,testEntry))
	print testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb)
	testEntry = ['stupid','garbage']
	thisDoc = array(setOfWord2Vec(myVocabList,testEntry))
	print testEntry,'classified as:',classifyNB(thisDoc,p0v,p1v,pAb)