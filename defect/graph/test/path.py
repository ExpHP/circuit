
import unittest
from defect.graph.path import *

class MiscTests(unittest.TestCase):
	def test_rotate_cycle(self):
		# a positive (right) rotation
		self.assertListEqual(rotate_cycle([4,5,6,7,8,4],  3), [7,8,4,5,6,7])
		# a negative (left) rotation
		self.assertListEqual(rotate_cycle([4,5,6,7,8,4], -1), [8,4,5,6,7,8])
		# a zero rotation (identity)
		self.assertListEqual(rotate_cycle([4,5,6,7,8,4],  0), [4,5,6,7,8,4])

	def test_can_join(self):
		# can be joined in given order
		assert can_join([1,2,3],[3,4,5])
		assert can_join([1,2,3],[3,4,5],swapok=True)

		# can be joined, but requires swapping order
		assert not can_join([3,4,5],[1,2,3])
		assert can_join([3,4,5],[1,2,3],swapok=True)

		# no matching endpoints
		assert not can_join([1,2,3],[7,8,9])
		assert not can_join([1,2,3],[7,8,9],swapok=True)

	def test_join(self):
		# can be joined
		self.assertListEqual(join([1,2,3],[3,4,5]),  [1,2,3,4,5])
		# cannot be joined
		self.assertRaises(ValueError, join, [3,4,5], [1,2,3])


