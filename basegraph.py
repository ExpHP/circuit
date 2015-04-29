
import random

import igraph

import traversal


# TODO: this currently implements some behavior specific to undirected graphs;
#       abstract this out with a subclass and/or mixin

class BaseGraph:

	# A bit of apps hungarian:
	#     v: a vertex id (as exposed to user, never changes)
	#     e: an edge id (as exposed to user, never changes)
	# rv,re: a 'raw' id (used by igraph, may change)

	# The purpose of these prefixes is to make errors easier to spot.
	# - methods on self._rg accept and return re and rv --- NEVER e and v.
	# - values that enter or leave this class are e and v --- NEVER re and rv.

	def __init__(self):
		self._rg = igraph.Graph()

		self._next_e = 0

		self._rv_map = {}
		self._v_map  = {}
		self._re_map = {}
		self._e_map  = {}

	# TODO: Implement these properly before exposing any 'delete' methods.
	def _v_from_rv(self, rv): return self._v_map[rv]
	def _rv_from_v(self, v):  return self._rv_map[v]
	def _e_from_re(self, re): return self._e_map[re]
	def _re_from_e(self, e):  return self._re_map[e]

	def add_vertices(self, vs):

		vs = list(vs)
		if len(vs) != len(set(vs)):
			raise ValueError('vertex {} specified multiple times'.format(repr(v)))

		for v in vs:
			if v in self._rv_map:
				raise ValueError('Vertex {} already in graph!'.format(repr(v)))

		# gather (expected) indices of new vertices
		rvFirst = self._rg.vcount()
		rvs = range(rvFirst, rvFirst + len(vs))

		v_rv_pairs = list(zip(vs,rvs))
		self._rv_map.update(
			{v:rv for (v,rv) in v_rv_pairs}
		)
		self._v_map.update(
			{rv:v for (v,rv) in v_rv_pairs}
		)

		self._rg.add_vertices(len(vs))

		assert(rvs.stop == self._rg.vcount())

	def _predict_new_res(self, n):
		reFirst = self._rg.ecount()
		return range(reFirst, reFirst + n)

	def _predict_new_rvs(self, n):
		rvFirst = self._rg.vcount()
		return range(rvFirst, rvFirst + n)

	def add_edge(self, v1, v2, **kwargs):
		rv1, rv2 = self._rv_from_v(v1), self._rv_from_v(v2)
		re, = self._predict_new_res(1)

		e = self._next_e
		self._next_e += 1

		self._rg.add_edge(rv1, rv2, **kwargs)

		self._re_map[e] = re
		self._e_map[re] = e

	def num_vertices(self):
		return self._rg.vcount()

	def num_edges(self):
		return self._rg.ecount()

	def vertices(self):
		return iter(self._rv_map)

	def edges(self):
		return iter(self._re_map)

	def _incident(self, v, mode):
		rv = self._rv_from_v(v)
		res = self._rg.incident(rv, mode=mode)
		# Make a collection now; We do NOT want lazy evaluation! (the raw indices could get invalidated)
		return tuple(map(self._e_from_re, res))

	def incident_edges(self, v):
		return self._incident(v, mode=igraph.ALL)
	def in_edges(self, v):
		return self._incident(v, mode=igraph.IN)
	def out_edges(self, v):
		return self._incident(v, mode=igraph.OUT)

	def arbitrary_edge(self, v1, v2):
		# TODO: error cases? (e.g. no such edge?)
		(rv1, rv2) = (self._rv_from_v(v1), self._rv_from_v(v2))
		re = self._rg.get_eid(rv1, rv2)
		return self._e_from_re(re)

	# Get the source and target vertices of an edge.
	def edge_endpoints(self, e):
		re = self._re_from_e(e)
		rvSource = self._rg.es[re].source
		rvTarget = self._rg.es[re].target

		return self._v_from_rv(rvSource), self._v_from_rv(rvTarget)

	# General method for obtaining the other vertex of an edge...
	#  but ONLY if the specified vertex is a valid source. (else, error)
	def edge_target_given_source(self, e, vSource):
		source,target = self.edge_endpoints(e)
		if   vSource == source:
			return target
		elif vSource == target: # FIXME directed should not allow this
			return source
		else:
			raise ValueError('vertex {} cannot be the source of edge'
				' {} (which connects {} to {})'.format(vSource,e,source,target))

	# General method for obtaining the other vertex of an edge...
	#  but ONLY if specified vertex is a valid target. (else, error)
	def edge_source_given_target(self, e, vTarget):
		source,target = self.edge_endpoints(e)
		if   vTarget == target:
			return source
		elif vTarget == source: # FIXME directed should not allow this
			return target
		else:
			raise ValueError('vertex {} cannot be the target of edge'
				' {} (which connects {} to {})'.format(vTarget,e,source,target))

	def cycle_basis(self):
		''' Generate and return a cycle basis for the graph as a list of paths.

		A cycle basis is a minimal-size set of cycles such that any cycle in the
		graph can be defined as the XOR sum of cycles in the basis.

		Note that, generally speaking, the cycle basis of a graph is NOT unique.'''

		# Algorithm:
		# - Make a spanning tree/forest. ANY tree will do.
		# - For each non-tree edge, traverse the tree between the endpoints to complete a cycle.

		# FIXME: This algo currently assumes the graph is all connected, and just generates
		#   a single rooted tree (instead of a forest)

		# get tree structure
		eBackEdges = {}
		vRoot = self.random_vertex()
		rvRoot = self._rv_from_v(vRoot)
		for (vobjNode, distance, vobjPred) in self._rg.bfsiter(rvRoot, mode=igraph.OUT, advanced=True):
			if vobjPred is None: continue
			vNode = self._v_from_rv(vobjNode.index)
			vPred = self._v_from_rv(vobjPred.index) if vobjPred is not None else None
			eBackEdges[vNode] = self.arbitrary_edge(vPred, vNode)

		# helper method - returns list of edges from root to v
		def path_from_root(v):
			result = []

			while v != vRoot:
				result.append(eBackEdges[v])
				v = self.edge_source_given_target(eBackEdges[v], v)

			result.reverse()
			return result

		# build a cycle from each non-tree edge
		cycles = []
		eTreeEdges = set(eBackEdges.values())
		for e in self.edges():
			if e not in eTreeEdges:

				# find paths from root to each node
				source, target = self.edge_endpoints(e)
				sourcePath = path_from_root(source)
				targetPath = path_from_root(target)

				# length of path to common ancestor
				iFrom = len(common_prefix(sourcePath, targetPath))

				cycle = [e]                                # source to target (via non-tree edge)
				cycle.extend(reversed(targetPath[iFrom:])) # target to ancestor (along tree)
				cycle.extend(sourcePath[iFrom:])           # ancestor to source (along tree)

				cycles.append(cycle)

		return cycles

	def random_vertex(self):
		print(list(self._rg.vs))
		rvCount = self._rg.vcount()
		rv = random.randrange(rvCount)
		return self._v_from_rv(rv)

	def _igraph_vertex(self, v):
		rv = self._rv_from_v(v)
		return self._rg.vs[rv]

	def _igraph_edge(self, e):
		re = self._re_from_e(e)
		return self._rg.es[re]

	def dfs_rooted(self, *args, **kwargs):
		return traversal.dfs_rooted(self, *args, **kwargs)

	def bfs_rooted(self, *args, **kwargs):
		return traversal.bfs_rooted(self, *args, **kwargs)

	def dfs_full(self, *args, **kwargs):
		return traversal.dfs_full(self, *args, **kwargs)

	def bfs_full(self, *args, **kwargs):
		return traversal.bfs_full(self, *args, **kwargs)


def common_prefix(*iterables):
	if len(iterables) == 0:
		raise ValueError('no iterables provided!')

	result = []
	for values in zip(*iterables):
		if any(v != values[0] for v in values[1:]):
			break
		result.append(values[0])

	return result

assert(common_prefix([0,'a',3,2,6,6,7], [0,'a',3,2,9,6]) == [0,'a',3,2])
assert(common_prefix([0,'a',3,2,6,6,7]) == [0,'a',3,2,6,6,7])

# test example
# TODO: remove or put elsewhere
def dfs_preorder_postorder(graph):
	preorder = []
	postorder = []
	graph.dfs_full(
		start_vertex = lambda g,v: preorder.append(v),
		discover_vertex = lambda g,v: preorder.append(v),
		finish_vertex = lambda g,v: postorder.append(v),
		)
	assert len(preorder) == len(postorder) == graph.num_vertices()
	return preorder,postorder

g = BaseGraph()
g.add_vertices([chr(ord('A')+i) for i in range(10)])
g.add_edge('A', 'B')
g.add_edge('A', 'H')
g.add_edge('B', 'C')
g.add_edge('B', 'E')
g.add_edge('B', 'I')
g.add_edge('C', 'F')
g.add_edge('D', 'F')
g.add_edge('D', 'G')
g.add_edge('E', 'F')
g.add_edge('F', 'G')
g.add_edge('F', 'I')
g.add_edge('F', 'I')
g.add_edge('G', 'J')
g.add_edge('I', 'J')
g.add_edge('F', 'H')

pre,post = dfs_preorder_postorder(g)
print(pre)
print(post)
