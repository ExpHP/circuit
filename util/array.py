
import numpy as np
import itertools

def axis_to_axeslist(axis, ndim):
	'''
	Taken from example on http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html .

	It takes an axis argument in the format accepted by numpy's various reduction functions
	(e.g. np.sum()), and returns a list in the format accepted by the ``op_axis`` argument
	of ``np.nditer`` (which appears to be a list of length ``ndim`` mapping each original
	axis to its new axis (or to -1 if said axis is reduced over).

	>>> axis_to_axeslist(None, 5) # default (reduce all)
	[-1, -1, -1, -1, -1]
	>>> axis_to_axeslist(2, 5) # scalar
	[0, 1, -1, 2, 3]
	>>> axis_to_axeslist((1, -2, 3), 6) # negative axes count backwards from end
	[0, -1, 1, -1, -1, 2]
	'''
	if axis is None:
		return [-1] * ndim
	else:
		if type(axis) is not tuple:
			axis = (axis,)
		axeslist = [1] * ndim
		for i in axis:
			axeslist[i] = -1
		ax = 0
		for i in range(ndim):
			if axeslist[i] != -1:
				axeslist[i] = ax
				ax += 1
		return axeslist


def smash_equal(arr, axis=None):
	'''
	Reduces an array of identical values to the first value.

	This is not the same as simply removing duplicates, as it is required that
	all elements are identical; otherwise, ``ValueError`` is raised.
	This is provided as a checked, fail-fast alternative to the practice of simply
	taking the first element of a sequence whose elements are assumed to be equal.

	``axis`` is in the same form accepted by ``numpy.sum``. When provided, only
	the elements which are actually being smashed together are required to be
	equal.

	>>> x = [[[0]*2, [1]*2]]*2 # values along axes 0 and 2 are identical
	>>> x = np.array(x)
	>>> smash_equal(x, axis=0)
	array([[0, 0],
	       [1, 1]])
	>>> smash_equal(x, axis=(0,2))
	array([0, 1])
	>>> smash_equal(x, axis=1)
	Traceback (most recent call last):
	  ...
	ValueError: Value mismatch!
	>>> smash_equal(x[:,0,:])
	array(0)
	'''
	if isinstance(arr, np.ma.masked_array):
		# proper support of these requires careful consideration
		raise TypeError('masked arrays are not supported')

	# FIXME not sure how numpy functions usually handle axes of 0 length
	if 0 in arr.shape:
		raise RuntimeError('Encountered axis of length 0!') # abort! abort!

	axeslist = axis_to_axeslist(axis, arr.ndim)
	outshape = tuple([arr.shape[i] for i,a in enumerate(axeslist) if a != -1])
	out = np.zeros(outshape, dtype=arr.dtype)

	# couldn't scale `numpy.nditer`'s learning cliff :/
	for out_idxs in itertools.product(*(range(n) for n in outshape)):
		in_idxs = tuple([slice(None) if a == -1 else out_idxs[a] for a in axeslist])
		elems = arr[in_idxs].flatten()

		out[out_idxs] = elems[0]
		if (elems != out[out_idxs]).any():
			raise ValueError('Value mismatch!')
	return out

