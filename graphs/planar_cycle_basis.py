
import enum
import math
import random
import itertools

import networkx as nx

from util import window2

__all__ = [
	'planar_cycle_basis',
	'planar_cycle_basis_nx',
	'without_vertex',
	'cyclebases_equal',
]

# MASSIVE TODO: planar_cycle_basis_nx appears to be very buggy and
#  unreliable right now, especially once many vertices are removed
#  from the bridge.  Its role has been reduced to just generating
#  the initial cyclebasis for now, but at some point I gotta get
#  in there with a debugger and figure out what's up

# vs:   iterable(V)
# es:   {E: (V, V)}
# v_xs: {V: float}
# v_ys: {V: float}
def planar_cycle_basis(vs, es, v_xs, v_ys):
	my_g = nx.Graph()
	my_g.add_nodes_from(vs)
	my_g.add_edges_from(es.values())
	nx.set_node_attributes(my_g, 'x', v_xs)
	nx.set_node_attributes(my_g, 'y', v_ys)

	return planar_cycle_basis_impl(my_g)

def planar_cycle_basis_nx(g, xs, ys):
	if g.is_directed():
		raise TypeError('Not implemented for directed graphs')
	if g.is_multigraph():
		raise TypeError('Not implemented for multi-graphs')

	if not (set(g) == set(xs) == set(ys)):
		raise ValueError('g, xs, ys must all share same set of vertices')

	my_g = nx.Graph()
	my_g.add_nodes_from(iter(g))
	my_g.add_edges_from(g.edges())

	nx.set_node_attributes(my_g, 'x', xs)
	nx.set_node_attributes(my_g, 'y', ys)

	return planar_cycle_basis_impl(my_g)


def show_g(g, cycle=None, with_labels=False):
	import matplotlib.pyplot as plt
	xs = nx.get_node_attributes(g,'x')
	ys = nx.get_node_attributes(g,'y')
	pos = {v:(xs[v],ys[v]) for v in xs}
	fig, ax = plt.subplots()
	nx.draw_networkx(g, with_labels=with_labels, ax=ax, pos=pos, nodelist=[])
	if cycle is not None:
		edges = list(path_edges(cycle))
		nx.draw_networkx(g, ax=ax, pos=pos, with_labels=False, nodelist=[cycle[0]], edgelist=edges, edge_color='r', width=2, node_size=5)
	plt.show()


def edge_angle(g, s, t):
	s_attr, t_attr = g.node[s], g.node[t]
	return math.atan2(t_attr['y']-s_attr['y'], t_attr['x']-s_attr['x'])


# Takes a mutable graph g with the following attributes:
#  vertex props:
#    x
#    y
# WARNING: The graph will be mutated.
def planar_cycle_basis_impl(g):
	assert not g.is_directed()
	assert not g.is_multigraph()

	# "First" nodes/edges for each cycle are chosen in an order such that the first edge
	#  may never belong to a later cycle.

	# rotate to (hopefully) break any ties or near-ties along the y axis
	rotate_graph(g, random.random() * 2 * math.pi)

	# NOTE: may want to verify that no ties exist after the rotation

	nx.set_edge_attributes(g, 'used_once', {e: False for e in g.edges()})

	# precompute edge angles
	angles = {}
	for s,t in g.edges():
		angles[s,t] = edge_angle(g,s,t) % (2*math.pi)
		angles[t,s] = (angles[s,t] + math.pi) % (2*math.pi)

	# identify and clear away edges which cannot belong to cycles
	for v in g:
		if degree(g,v) == 1:
			remove_filament_from_tip(g, v)

	cycles = []

	# sort ascendingly by y
	for root in sorted(g, key = lambda v: g.node[v]['y']):

		# Check edges in ccw order from the +x axis
		for target in sorted(g[root], key=lambda t: angles[root,t]):

			if not g.has_edge(root, target):
				continue

			discriminator, path = trace_ccw_path(g, root, target, angles)

			if discriminator == PATH_PLANAR_CYCLE:
				assert path[0] == path[-1]
				remove_cycle_edges(g, path)
				cycles.append(path)

			# Both the dead end and the initial edge belong to filaments
			elif discriminator == PATH_DEAD_END:
				import pdb;pdb.set_trace()
				remove_filament_from_edge(g, root, target)
				remove_filament_from_tip(g, path[-1])

			# The initial edge must be part of a filament
			elif discriminator == PATH_OTHER_CYCLE:
				import pdb;pdb.set_trace()
				remove_filament_from_edge(g, root, target)

			else: assert False # complete switch

			assert not g.has_edge(root, target)

	assert len(g.edges()) == 0
	return cycles

# g.degree is a tad slow
def degree(g, v):
	return len(g.edge[v])

def rotate_graph(g, angle):
	xs = nx.get_node_attributes(g, 'x')
	ys = nx.get_node_attributes(g, 'y')

	xs,ys = rotate_coord_maps(xs, ys, angle)

	nx.set_node_attributes(g, 'x', xs)
	nx.set_node_attributes(g, 'y', ys)

def rotate_coord_maps(xs, ys, angle):
	sin,cos = math.sin(angle), math.cos(angle)
	newx = {v: cos*xs[v] - sin*ys[v] for v in xs}
	newy = {v: sin*xs[v] + cos*ys[v] for v in xs}
	return newx,newy


# Attempts to trace a planar cycle by following the contour counterclockwise.
# May not necessarily find a good cycle.
PATH_DEAD_END     = 'dead end'
PATH_PLANAR_CYCLE = 'cycle'
PATH_OTHER_CYCLE  = 'revisited'
def trace_ccw_path(g, vfirst, vsecond, angles):
	prev, cur = vfirst, vsecond
	path = [vfirst]
	visited = set(path)

	def interior_angle_cw(v1, v2, v3):
		return (angles[v2,v1] - angles[v2,v3]) % (2*math.pi)

	while cur not in visited:

		assert g.has_edge(path[-1],cur)
		path.append(cur)
		visited.add(cur)

		# find edge with smallest clockwise interior angle...
		neighbor_it = iter(sorted(g[cur], key = lambda t: interior_angle_cw(prev, cur, t)))
		# ... excluding the edge that brought us here
		neighbor_it = itertools.dropwhile(lambda t: t == path[-2], neighbor_it)

		prev = cur
		cur  = next(neighbor_it, None)

		if cur is None:
			# no edges except the way we came;
			assert path[-1] == prev
			return (PATH_DEAD_END, path)

	assert cur in visited
	path.append(cur)

	if cur == vfirst:
		# made it back to the start: planar cycle
		return (PATH_PLANAR_CYCLE, path)
	else:
		# found another previous vertex: cycle of unknown quality
		return (PATH_OTHER_CYCLE, path)
	assert False

def remove_cycle_edges(g, path):
	assert len(path) == len(set(path)) + 1

	# delete any edge belonging to two cycles
	for prev, cur in window2(path):
		if g.edge[prev][cur]['used_once']: g.remove_edge(prev,cur)
		else: g.edge[prev][cur]['used_once'] = True

	# always delete first edge of cycle
	if g.has_edge(path[0], path[1]):
		g.remove_edge(path[0], path[1])

	for v in path:
		if degree(g,v) == 1:
			remove_filament_from_tip(g,v)

# deletes a chain of sequentially linked edges given one of them
def remove_filament_from_edge(g,s,t):
	g.remove_edge(s, t)
	for v in (s,t):
		if degree(g,v) == 1:
			remove_filament_from_tip(g,v)

# deletes an orphaned chain of edges, given the vertex at its end
def remove_filament_from_tip(g,v):
	assert degree(g,v) == 1
	while degree(g,v) == 1:
		neighbor = next(iter(g.edge[v]))
		g.remove_edge(v, neighbor)
		v = neighbor


path_edges = window2

def test_known_cyclebasis(g, xs, ys, expected):
	print('ex',canonicalize_cyclebasis(expected))
	for step in range(10):
		rotx, roty = rotate_coord_maps(xs, ys, 2*math.pi*random.random())
		cb = planar_cycle_basis_nx(g, rotx, roty)

		print('cb',canonicalize_cyclebasis(cb))
		assert cyclebases_equal(cb, expected)

# a cycle inside another, sharing a vertex
def test_triangles():
	g = nx.Graph()
	g.add_path([0,1,2,0,3,4,0])
	xs,ys = {}, {}
	xs[0] =  0.0; ys[0] = 0.0
	xs[1] = -0.5; ys[1] = 1.0
	xs[2] = +0.5; ys[2] = 1.0
	xs[3] = -1.5; ys[3] = 2.0
	xs[4] = +1.5; ys[4] = 2.0

	test_known_cyclebasis(g, xs, ys, [[0,1,2,0],[0,3,4,0]])

# a cycle inside another, connected by a 1-edge filament
def test_hanging_diamond():
	g = nx.Graph()
	g.add_path([0,1,2,3,0,4,5,6,7,4])
	xs,ys = {v:0.0 for v in g}, {v:0.0 for v in g}

	xs[1] = +1.0; xs[5] = +0.5
	xs[3] = -1.0; xs[7] = -0.5
	ys[0] = +1.0; ys[4] = +0.5
	ys[2] = -1.0; ys[6] = -0.5

	test_known_cyclebasis(g, xs, ys, [[0,1,2,3,0],[4,5,6,7,4]])

#----------------------------------------------------

def cyclebases_equal(a, b):
	if len(a) != len(b):
		return False

	a_vertices = set(itertools.chain(*a))
	b_vertices = set(itertools.chain(*b))

	if a_vertices != b_vertices:
		return False

	vertex_ids = {v:i for i,v in enumerate(a_vertices)}
	a_canonical = canonicalize_cyclebasis(a, key=vertex_ids.__getitem__)
	b_canonical = canonicalize_cyclebasis(a, key=vertex_ids.__getitem__)

	return a_canonical == b_canonical

def canonicalize_cyclebasis(cyclebasis, key=lambda v:v):
	return sorted(canonicalize_cycle(c,key) for c in cyclebasis)

def canonicalize_cycle(seq, key=lambda x:x):
	seq = tuple(map(key, seq))

	assert seq[0] == seq[-1] # code dependency on precise format of cycles

	# Rotate to place minimum element first
	seq = rotate_cycle(seq, seq.index(min(seq)))

	# Flip direction to minimize second element
	if seq[-2] < seq[1]:
		seq = seq[::-1]

	return seq

#----------------------------------------------------

# Updates a cyclebasis for a straight-edge planar graph to account for the
#  removal of the given vertex
def without_vertex(cyclebasis, v):

	invalidated, unchanged = partition(lambda c: v in c, cyclebasis)

	segments = [break_cycle_at(c, v) for c in invalidated]

	segments = align_paths(segments)
	segments = recombine_aligned_paths(segments)
	newcycles = list(filter(is_cycle, segments))
	assert len(newcycles) < 2 # straight-edge planar graph should not form more than 1 new cycle

	newcycles = [simplify_retraced_edges(c) for c in newcycles]

	newcycles.extend(list(c) for c in unchanged)
	return newcycles

# Return the path formed by removing a vertex from a cycle
def break_cycle_at(cycle, v):
	cycle = rotate_cycle(cycle, cycle.index(v))
	assert cycle[0] == cycle[-1] == v
	return cycle[1:-1]

def rotate_cycle(cycle, n):
	assert cycle[0] == cycle[-1] # don't replace with is_cycle
	                             # (this method is dependent on the precise format of cycles)
	n %= (len(cycle) - 1)
	return cycle[n:] + cycle[1:n+1]

assert rotate_cycle([4,5,6,7,8,4],3)  == [7,8,4,5,6,7]
assert rotate_cycle([4,5,6,7,8,4],-1) == [8,4,5,6,7,8]
assert rotate_cycle([4,5,6,7,8,4],0)  == [4,5,6,7,8,4]

assert break_cycle_at([4,5,6,7,8,4],6) == [7,8,4,5]
assert break_cycle_at([4,5,6,7,8,4],4) == [5,6,7,8]

def partition(pred, it):
	yes,no = [],[]
	for x in it:
		if pred(x): yes.append(x)
		else:       no.append(x)
	return yes,no

assert partition(lambda x: x%3==0, range(7)) == ([0,3,6],[1,2,4,5])

# A list of paths are "aligned" if no two paths share the same first vertex
#  or the same last vertex.
# By joining them at matching ends, there exists a unique smallest set of paths
#  into which a set of aligned paths can be recombined. (unique up to rotations
#  of any cycles, at least)
def are_aligned(paths):
	firsts = set(p[0] for p in paths)
	lasts  = set(p[-1] for p in paths)
	return len(firsts) == len(lasts) == len(paths)

assert     are_aligned([[1,2],[2,1]]) # two shared endpoints between two edges is OK
assert     are_aligned([[1,2,3],[6,7,1],[3,4,5]])
assert not are_aligned([[1,2,3],[6,7,1],[5,4,3]]) # duplicate end (3)
assert not are_aligned([[1,2,3],[1,7,6],[3,4,5]]) # duplicate first (1)

def is_cycle(path):
	return path[0] == path[-1]

# Returns an aligned set of paths made by reversing some of the paths.
# It is an error if this is not possible (e.g. if 3 paths share an endpoint)
def align_paths(paths):
	paths = list(paths)

	if len(paths) == 0:
		return []

	result = []
	firsts_taken = set()
	lasts_taken  = set()

	def add_to_result(path):
		assert path[0] not in firsts_taken
		assert path[-1] not in lasts_taken
		result.append(path)
		firsts_taken.add(path[0])
		lasts_taken.add(path[-1])

	# constraints imposed by paths already in result
	def must_flip(path):
		return path[0] in firsts_taken or path[-1] in lasts_taken
	def must_not_flip(path):
		return path[0] in lasts_taken or path[-1] in firsts_taken

	remaining = set(range(len(paths)))

	# start with an arbitrary path
	add_to_result(paths[remaining.pop()])

	while len(remaining) > 0:
		constrained = set()

		# find paths constrained by paths in result
		for i in remaining:
			path = paths[i]

			if must_flip(path):

				path = path[::-1]
				if must_flip(path):  # Neither orientation meets constraints!
					raise ValueError('Paths cannot be aligned!')

				assert must_not_flip(path)  # the `if` version of "// fallthrough" :D

			if must_not_flip(path):
				add_to_result(path)
				constrained.add(i)

		remaining.difference_update(constrained)

		# nothing is constrained; add an arbitrary path to get things moving again
		if len(constrained) == 0:
			add_to_result(paths[remaining.pop()])

	assert are_aligned(paths)
	return paths

def get_first(path, flip):
	return path[-1 if flip else 0]

assert get_first([1,2,3,4],False) == 1
assert get_first([1,2,3,4],True)  == 4

def recombine_aligned_paths(paths):
	assert are_aligned(paths) # precondition

	# keep going until no joins can be made
	again = True
	while again:
		again = False
		new_paths = []

		for oldI,old in enumerate(paths):

			# try sticking 'old' onto one of the 'new' paths
			for newI,new in enumerate(new_paths):

				if can_join_vertex_paths(old, new):
					new_paths[newI] = join_vertex_paths(old, new)
					again = True
					break

			# 'old' could not be joined onto an existing path
			else:
				new_paths.append(old)

		assert len(new_paths) <= len(paths)
		assert (len(new_paths) < len(paths)) == again
		paths = new_paths

	return paths

def can_join_vertex_paths(a, b):
	return a[0] == b[-1] or b[0] == a[-1]

def join_vertex_paths(a, b):
	# use common vertex to determine order of paths
	if a[-1] != b[0]:
		a,b = b,a
	if a[-1] != b[0]:
		raise ValueError('Paths cannot be joined at either end!')

	tmp = a[:-1] # exclude common vertex
	tmp.extend(b)
	return tmp

# TODO: there's probably another better known algorithm for this
def simplify_retraced_edges(cycle):
	# use a stack to identify retraced sequences

	# begin with a copy of the cycle already on the stack, so we can
	#  detect retraced sequences that wrap around the list ends
	stack = cycle[:-1]

	# parameters of result
	offset = len(stack)
	length = len(cycle) - 1

	for v in cycle[:-1]:
		# check against SECOND to last; ABA should become A
		if stack[-2] == v:
			stack.pop()
			length -= 2
		else:
			stack.append(v)

		offset = min(offset, len(stack))

	result = stack[offset : offset+length]
	result.append(result[0]) # make into cycle
	return result

# TODO: there's probably another better known algorithm for this
def simplify_retraced_edges(cycle):
	# use a stack to identify retraced sequences
	stack = []

	for v in cycle:
		# check against SECOND to last; ABA should become A
		if len(stack) >= 2 and stack[-2] == v:
			stack.pop()
		else:
			stack.append(v)

	# eliminate unnecessary repeated vertices from end
	# e.g. ABCX...YCBA -> CX...YC
	k = 0
	while stack[k+1] == stack[-k-2]:
		k += 1
	if k > 0:
		stack = stack[k:-k]

	return stack

def cycle_permutations(c):
	c = list(c)
	for i in range(len(c)-1):
		yield rotate_cycle(c,i)

# check permutations to verify it catches stuff wrapping around the end
for c in cycle_permutations([3,50,70,3,2,1,0,-1,-2,-1,0,1,2,3]):
	assert simplify_retraced_edges(c) in cycle_permutations([3,50,70,3])

# struggling to understand why unittest doesn't offer its useful
#  assertions as free functions
def assertRaises(cls, f, *args, **kw):
	try:
		f(*args,**kw)
	except cls:
		return
	assert False

assert     can_join_vertex_paths([1,2,3],[3,4,5])
assert     can_join_vertex_paths([3,4,5],[1,2,3])
assert not can_join_vertex_paths([1,2,3],[7,8,9])

assert join_vertex_paths([1,2],[2,3,4]) == [1,2,3,4]
assert join_vertex_paths([2,3,4],[1,2]) == [1,2,3,4]
assertRaises(ValueError, join_vertex_paths, [1,2], [3,4,5])

assert recombine_aligned_paths([[1,2,3],[4,5,6]]) == [[1,2,3],[4,5,6]] # bad-ish test; output order is arbitrary
assert recombine_aligned_paths([[1,2,3],[5,6,7],[3,4,5]]) == [[1,2,3,4,5,6,7]]
assert is_cycle(recombine_aligned_paths([[1,2,3],[5,6,1],[3,4,5]]))

test_triangles()
test_inner_diamond()
import sys;sys.exit(0)
