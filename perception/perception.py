# -*- coding: utf-8 -*-
import copy
from matplotlib import pyplot as plt
from matplotlib import animation

training_set = [[(3,3),1],[(4,3),1],[(1,1),-1]]
w = [0,0]
b = 0
history = []

"""
使用随机梯度下降方法更新参数
"""
def update(item):

	global w,b,history
	w[0] += 1 * item[1] * item[0][0]
	w[1] += 1 * item[1] * item[0][1]
	b += 1 * item[1]
	print(w,b)
	history.append([copy.copy(w),b])

"""
计算item到超平面s的距离，输出yi(w*xi+b)
"""
def cal(item):

	res = 0
	for i in range(len(item[0])):
		res += item[0][i] * w[i]
	res += b
	res *= item[1]
	return res

"""
检查超平面是否能够成功分类
"""
def check():
	flag = False
	for item in training_set:
		if cal(item) <= 0:
			flag = True
			update(item)
	if not flag:
		print("RESULT: w: " + str(w) + " b: " + str(b))
	return flag

if __name__ == "__main__":
	for i in range(1000):
		if not check():
			break
