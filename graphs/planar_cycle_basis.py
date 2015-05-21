
import enum
import math
import random
import itertools

import networkx as nx

# vs:   iterable(V)
# es:   {E: (V, V)}
# v_xs: {V: float}
# v_ys: {V: float}
def minimal_cycle_basis(vs, es, v_xs, v_ys):
	my_g = nx.Graph()
	my_g.add_nodes_from(vs)
	my_g.add_edges_from(es.values())
	nx.set_node_attributes(my_g, 'x', v_xs)
	nx.set_node_attributes(my_g, 'y', v_ys)

	return minimal_cycle_basis_impl(my_g)

def minimal_cycle_basis_nx(g, xs, ys):
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

	return minimal_cycle_basis_impl(my_g)


@enum.unique
class EdgeState(enum.Enum):
	UNUSED    = 0 # an edge part of no cycles
	USED_ONCE = 1 # an edge part of only one cycle
	DELETED   = 2 # an edge which may not belong to any more cycles

def edge_angle(g, s, t):
	s_attr, t_attr = g.node[s], g.node[t]
	return math.atan2(t_attr['y']-s_attr['y'], t_attr['x']-s_attr['x'])


# Takes a mutable graph g with the following attributes:
#  vertex props:
#    x
#    y
#  edge props:
#    ...
def minimal_cycle_basis_impl(g):
	assert not g.is_directed()
	assert not g.is_multigraph()

	nx.set_edge_attributes(g, 'state', {e: EdgeState.UNUSED for e in g.edges()})

	cycles = []

	# "First" nodes/edges for each cycle are chosen in an order such that the first edge
	#  may never belong to a later cycle.

	# rotate to (hopefully) break any ties or near-ties along the y axis
	rotate_graph(g, random.random() * 2 * math.pi)

	# NOTE: may want to verify that no ties exist after the rotation

	# sort ascendingly by y
	for root in sorted(g, key = lambda v: g.node[v]['y']):

		# Check edges in ccw order from the +x axis
		for target in sorted(g[root], key=lambda t: edge_angle(g,root,t) % (2*math.pi)):

			if not g.has_edge(root, target):
				continue

			if g.edge[root][target]['state'] is EdgeState.DELETED:
				continue

			path = maybe_complete_cw_cycle(g, root, target)
			if path is None:
				continue

			mark_cycle_edges(g, path)

			cycles.append(path)

	return cycles

def rotate_graph(g, angle):
	xs = nx.get_node_attributes(g, 'x')
	ys = nx.get_node_attributes(g, 'y')

	sin,cos = math.sin(angle), math.cos(angle)
	for v in g:
		x, y = xs[v], ys[v]
		xs[v] = cos*x - sin*y
		ys[v] = sin*x + cos*y

	nx.set_node_attributes(g, 'x', xs)
	nx.set_node_attributes(g, 'y', ys)

def maybe_complete_cw_cycle(g, vfirst, vsecond):
	prev, cur = vfirst, vsecond
	path = [vfirst, vsecond]

	assert g.edge[path[-2]][path[-1]]['state'] is not EdgeState.DELETED

	def interior_angle_cw(v1, v2, v3):
		return (edge_angle(g, v2, v1) - edge_angle(g, v2, v3)) % (2*math.pi)

	def bad_edge(v1, v2):
		return (
			(g.edge[v1][v2]['state'] is EdgeState.DELETED)
			or (v2 == path[-2]) # this is not redundant (think lists of length 2)
			or (v2 in path[1:])
		)

	while cur != vfirst:
		neighbor_it = iter(sorted(g[cur], key = lambda t: interior_angle_cw(prev, cur, t)))
		neighbor_it = itertools.dropwhile(lambda t: bad_edge(cur, t), neighbor_it)

		target = next(neighbor_it, None)

		if target is None: # no good edges; dead end
			return None

		prev = cur
		cur = target

		path.append(target)
		assert g.edge[path[-2]][path[-1]]['state'] is not EdgeState.DELETED

	assert path[0] == path[-1]
	return path

def mark_cycle_edges(g, path):
	assert len(path) == len(set(path)) + 1
	for prev, cur in window2(path):
		edict = g.edge[prev][cur]
		if   edict['state'] is EdgeState.UNUSED:    edict['state'] = EdgeState.USED_ONCE
		elif edict['state'] is EdgeState.USED_ONCE: edict['state'] = EdgeState.DELETED
		elif edict['state'] is EdgeState.DELETED:   assert False

	root, second = path[:2]
	g.edge[root][second]['state'] = EdgeState.DELETED

	for prev, cur in window2(path):
		if g.has_edge(prev,cur) and g.edge[prev][cur]['state'] is EdgeState.DELETED:
			delete_edge_and_filaments(g,prev,cur)

def delete_edge_and_filaments(g,s,t):
	g.remove_edge(s,t)
	for v in (s,t):
		while g.degree(v) == 1:
			neighbor = next(iter(g.edge[v]))
			g.remove_edge(v, neighbor)
			v = neighbor

# A scrolling 2-element window on an iterator
def window2(it):
	it = iter(it)
	prev = next(it)
	for x in it:
		yield (prev,x)
		prev = x

