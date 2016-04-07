from __future__ import print_function

import operator
import math
from collections import deque
from functools import wraps

COMPARE_CHILD = {
	0: (operator.le,operator.sub),
	1: (operator.ge,operator.add),
}

class Node(object):
	def __init__(self,data=None,left=None,right=None):
		self.data = data
		self.left = left
		self.right = right

	@property
	def is_leaf(self):
		return (not self.data) or \
				(all(not bool(c) for c, p in self.children))

	def preorder(self):
		if not self:
			return
		
		yield self

		if self.left:
			for x in self.left.preorder():
				yield x

		if self.right:
			for x in self.right.preorder():
				yield x

	def inorder(self):
		if not self:
			return

			if self.left:
				for x in self.left.inorder():
					yield x

			yield self

			if self.right:
				for x in self.right.inorder():
					yield x
	def postorder(self):
		if not self:
			return
		if self.left:
			for x in self.left.postorder():
				yield x
		if self.right:
			for x in self.right.postorder:
				yield x
		yield self

		
				