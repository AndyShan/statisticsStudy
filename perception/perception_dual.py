import numpy as np
"""
感知机算法的对偶形式
"""
training_set = np.array([[[3,3],1],[[4,3],1],[[1,1],-1]])
a = np.zeros(len(training_set),np.float)#长度为训练集大小的元素为0的参数a数组

b = 0.0#参数b，初始值为0
Gram = None#Gram矩阵
y = np.array(training_set[:,1])#使y数组等于训练集中的y
x = np.empty((len(training_set),2),np.float)
for i in range(len(training_set)):#使x数组等于训练集中的x
	x[i] = training_set[i][0]
history = []

"""
计算gram矩阵
"""
def cal_gram():
	g = np.empty((len(training_set),len(training_set)),np.int)
	for i in range(len(training_set)):
		for j in range(len(training_set)):
			g[i][j] = np.dot(training_set[i][0],training_set[j][0])#调用numpy的dot方法进行矩阵内积计算
	return g
"""
使用随机梯度下降进行参数更新
"""
def update(i):
	global a,b
	a[i] += 1
	b = b + y[i]
	history.append([np.dot(a * y,x),b])
	print(np.dot(a * y,x),b)

	
def cal(i):
	global a,b,x,y
	res = np.dot(a * y,Gram[i])
	res = (res + b) * y[i]
	return res

def check():
	global a,b,x,y
	flag = False
	for i in range(len(training_set)):
		if cal(i) <= 0:
			flag = True
			update(i)
	if not flag:
		w = np.dot(a * y,x)
		print("RESULT: w:" + str(w) + "b:" + str(b))
		return False
	return True

if __name__ == "__main__":
	print(len(training_set))
	Gram = cal_gram()
	for i in range(1000):
		if not check():
			break