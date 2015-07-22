# module with helper functions for dealing with vertex paths for non-multigraphs,
#  stored as lists of vertices.

# Also deals with cycles, stored as paths whose first vertex == their last.

import itertools

from .. import util

def can_join(a, b, swapok=False):
	return a[-1] == b[0] or (swapok and b[-1] == a[0])

def join(a, b, swapok=False):
	# use common vertex to determine order of paths
	if swapok and (a[-1] != b[0]):
		a,b = b,a
	if a[-1] != b[0]:
		raise ValueError('Paths cannot be joined!')

	tmp = a[:-1] # exclude common vertex
	tmp.extend(b)
	return tmp

def is_cycle(path):
	return path[0] == path[-1]

def rotate_cycle(cycle, n):
	assert cycle[0] == cycle[-1] # don't replace with is_cycle
	                             # (this method is dependent on the precise format of cycles)
	n %= (len(cycle) - 1)
	return cycle[n:] + cycle[1:n+1]

def cyclebases_equal(a, b, directed=False):
	if len(a) != len(b):
		return False

	a_vertices = set(itertools.chain(*a))
	b_vertices = set(itertools.chain(*b))

	if a_vertices != b_vertices:
		return False

	vertex_ids = {v:i for i,v in enumerate(a_vertices)} # guarantee strict total ordering
	a_canonical = canonicalize_cyclebasis(a, directed, key=vertex_ids.__getitem__)
	b_canonical = canonicalize_cyclebasis(a, directed, key=vertex_ids.__getitem__)

	return a_canonical == b_canonical


# canonicalization methods to convert objects into a more easily compared form,
# dealing away with issues such as the arbitrary starting point of a cycle, and etc.

# For best results, the graph vertices must be of a comparable type which forms a strict
#  total ordering (i.e. for all a, b, exactly one of the following is true: a<b, a==b, a>b).
# Not all hashable types are strictly ordered (int, string, and tuple are; frozenset is not).

# If necessary, you may provide a key function to map vertices to a strictly ordered type.

def canonicalize_cyclebasis(cyclebasis, directed=False, key=lambda v:v):
	return sorted(canonicalize_cycle(c,directed,key) for c in cyclebasis)


def canonicalize_cycle(seq, directed=False, key=lambda v:v):
	seq = tuple(map(key, seq))

	assert seq[0] == seq[-1] # code dependency on precise format of cycles

	# Rotate to place minimum element first
	seq = rotate_cycle(seq, seq.index(min(seq)))

	# Flip direction to minimize second element (undirected only)
	if (not directed) and seq[-2] < seq[1]:
		seq = seq[::-1]

	return seq

# edges(path) => iterator of (source,target) for each edge
edges = util.window2

#------------------------------------------------------------------

# /me needs to figure out how to write nose tests

assert rotate_cycle([4,5,6,7,8,4],3)  == [7,8,4,5,6,7]
assert rotate_cycle([4,5,6,7,8,4],-1) == [8,4,5,6,7,8]
assert rotate_cycle([4,5,6,7,8,4],0)  == [4,5,6,7,8,4]

assert     can_join([1,2,3],[3,4,5])
assert     can_join([3,4,5],[1,2,3],swapok=True)
assert not can_join([1,2,3],[7,8,9],swapok=True)
assert not can_join([3,4,5],[1,2,3])

assert join([1,2,3],[3,4,5]) == [1,2,3,4,5]
util.assertRaises(ValueError, join, [3,4,5], [1,2,3])

