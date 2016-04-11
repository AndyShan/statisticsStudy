# -*- coding: utf-8 -*-
import numpy

def kdtree(data leafsize = 10):
	
	ndim = data.shape[0] #数据的行数
	ndata = data.shape[1] #列数

# 查找边界超矩形
	hrect = numpy.zeros((2,data.shape[0]))
	hrect[0,:] = data.min(axis = 1) # 将hrect的第一行赋data矩阵中每行的最小元素
	hrect[1,:] = data.max(axis = 1) # 将hrect的第二行赋data矩阵中每行的最大元素

# 构造kd树的根节点
	idx = numpy.argsort(data[0,:],kind = 'mergesort') # idx为data第一行数据排序后结果的下标
	data[:,:] = data[:,idx]
	splitval = data[0,ndata/2]

	left_hrect = hrect.copy()
	right_hrect = hrect.copy()
	left_hrect[1,0] = splitval
	right_hrect[0,0] = splitval

	tree = [(None,None,left_hrect,right_hrect,None,None)]

	stack = [(data[:,:ndata/2],idx[:ndata/2],1,0,True),
			(data[:,ndata/2:],idx[ndata/2:],1,0,False)]

	while stack:
		
		data,didx,depth,parent,leftbranch = stack.pop()
		ndata = data.shape[1]
		nodeptr = len(tree)

		_didx,_data,_left_hrect,_right_hrect,left,right = tree[parent]

		tree[parent]  = (_didx,_data,_left_hrect,_right_hrect,nodeptr,right) if leftbranch else (_didx,_data,_left_hrect,_right_hrect,left,nodeptr)

		if ndata <= leafsize:
			_didx = didx.copy()
			_data = data.copy()
			leaf = (_didx,_data,None,None,0,0)
			tree.append(leaf)

		else:
			splitdim = depth % ndim
			idx = argsort(data[splitdim,:],kind = "mergesort")
			data[:,:] = data[:,idx]
			didx = didx[idx]
			nodeptr = len(tree)
			stack.append((data[:,:ndata/2],didx[:ndata/2],depth+1,nodeptr,True))
			stack.append((data[:,ndata/2:],didx[ndata/2:],depth+1,nodeptr,False))
			splitval = data[splitdim,ndata/2]
			if leftbranch:
				left_hrect = _left_hrect.copy()
				right_hrect = _left_hrect.copy()
			else:
				left_hrect = right_hrect.copy()
				right_hrect = right_hrect.copy()
			left_hrect[1,splitdim] = splitval
			right_hrect[0,splitdim] = splitval
			tree.append((None,None,left_hrect,right_hrect,None,None))

	return tree