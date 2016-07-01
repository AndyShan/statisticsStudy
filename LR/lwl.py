#  -*- coding: utf-8 -*-
from numpy import *
def loadFile(fileName):
    """
    读取数据
    :param fileName:文件路径
    :return:以数组形式返回训练数据中的x和y
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArray = [];
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArray.append(float(curLine[i]))
        dataMat.append(lineArray)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegres(xArr,yArr):
    """
    线性回归
    :param xArr:
    :param yArr:
    :return: 使成本函数最小的参数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T#将y矩阵转置
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:#求行列式
         print 'This matrix is singular, cannot do inverse'
         return
    ws = xTx.I * (xMat.T * yMat)#根据线性回归公式计算使成本函数最小的参数
    return ws

def plotStandRegres(xArr,yArr,ws):
    """
    绘制线性回归图像
    :param xArr:
    :param yArr:
    :param ws:
    :return:
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i[1] for i in xArr],yArr,'ro')
    xCopy = xArr
    print type(xCopy)
    xCopy.sort()
    yHat = xCopy*ws
    ax.plot([i[1] for i in xCopy],yHat)
    plt.show()

def calcCorrcoef(xArr,yArr,ws):
    """
    计算相关度
    :param xArr:
    :param yArr:
    :param ws:
    :return:相关度
    """
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    return corrcoef(yHat.T,yMat)

def lwl(queryPoint,xArr,yArr,k):
    """
    局部加权回归
    :param queryPoint:查询点
    :param xArr:
    :param yArr:
    :param k:参数，为1时lwl变为线性回归。值越小，拟合越准确，值过小时出现过拟合。
    :return:通过局部加权回归法计算的查询点的预测值
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = queryPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:  # 求行列式
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return queryPoint * ws

    def lwlrTest(queryArray,xArr,yArr,k):
        """
        利用局部加权回归拟合曲线
        :param queryArray:
        :param xArr:
        :param yArr:
        :param k:
        :return: 通过局部加权回归求出的拟合曲线
        """
        m = shape(queryArray)[0]
        yHat = zeros(m)
        for i in range(m):
            yHat[i] = lwl(queryArray[i],xArr,yArr,k)
        return yHat

def lwlrTestPlot(xArr,yArr,k):
    """
    绘制局部加权回归法求出的曲线
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    import matplotlib.pyplot as plt
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwl(xCopy[i],xArr,yArr,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i[1] for i in xArr],yArr,'ro')
    ax.plot(xCopy,yHat)
    plt.show()

if __name__ == '__main__':
    xArr,yArr = loadFile("")
    lwlrTestPlot(xArr,yArr,0.1)
    # print lwl(1,xArr,yArr,1)