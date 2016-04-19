# -*- coding: utf-8 -*-
from numpy import *

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

"""将一个大字符串拆分为字符串列表"""
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\w*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1,26):
		wordList = textParse(open('C:/Users/AD/statisticsStudy/demo/emailBayes/email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('C:/Users/AD/statisticsStudy/demo/emailBayes/email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = range(50)
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWord2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
			errorCount += 1
			print docList[docIndex]
	print 'this error rate is :',float(errorCount) / len(testSet)
if __name__ == '__main__':
	spamTest()
	