
import enum
import math
import random
import itertools

import networkx as nx

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


def edge_angle(g, s, t):
	s_attr, t_attr = g.node[s], g.node[t]
	return math.atan2(t_attr['y']-s_attr['y'], t_attr['x']-s_attr['x'])


# Takes a mutable graph g with the following attributes:
#  vertex props:
#    x
#    y
#  edge props:
#    ...
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
			remove_filament(g, v)

	cycles = []

	# sort ascendingly by y
	for root in sorted(g, key = lambda v: g.node[v]['y']):

		# Check edges in ccw order from the +x axis
		for target in sorted(g[root], key=lambda t: angles[root,t]):

			if not g.has_edge(root, target):
				continue

			path = maybe_complete_cw_cycle(g, root, target, angles)
			if path is None:
				continue

			remove_cycle_edges(g, path)

			cycles.append(path)

	return cycles

# g.degree is a tad slow
def degree(g, v):
	return len(g.edge[v])

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

def maybe_complete_cw_cycle(g, vfirst, vsecond, angles):
	prev, cur = vfirst, vsecond
	path = [vfirst, vsecond]
	forbidden = set([vsecond]) # non-root path nodes (which may not be revisited)

	def interior_angle_cw(v1, v2, v3):
		return (angles[v2,v1] - angles[v2,v3]) % (2*math.pi)

	def bad_edge(v1, v2):
		return v2 == path[-2] or v2 in forbidden

	while cur != vfirst:
		neighbor_it = iter(sorted(g[cur], key = lambda t: interior_angle_cw(prev, cur, t)))
		neighbor_it = itertools.dropwhile(lambda t: bad_edge(cur, t), neighbor_it)

		target = next(neighbor_it, None)

		if target is None: # no good edges; dead end
			return None

		prev = cur
		cur = target

		path.append(target)
		forbidden.add(target)
		assert g.has_edge(path[-2],path[-1])

	assert path[0] == path[-1]
	return path

def remove_cycle_edges(g, path):
	assert len(path) == len(set(path)) + 1

	# delete any edge belonging to two cycles
	for prev, cur in window2(path):
		if g.edge[prev][cur]['used_once']: g.remove_edge(prev,cur)
		else: g.edge[prev][cur]['used_once'] = True

	# always delete first edge of cycle
	if g.has_edge(path[0], path[1]):
		g.remove_edge(path[0], path[1])

	# clear away any filaments left behind
	for v in path:
		if degree(g,v) == 1:
			remove_filament(g,v)

# deletes an orphaned chain of edges, given the vertex at its end
def remove_filament(g,v):
	assert degree(g,v) == 1
	while degree(g,v) == 1:
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

