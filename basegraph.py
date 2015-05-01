
import random

import igraph

import traversal

class BaseGraph:

	# names of igraph attributes used to store fixed vertex/edge ids
	V_ATTRIBUTE = 'vlabel'
	E_ATTRIBUTE = 'elabel'

	# A bit of apps hungarian:
	#     v: a vertex id (as exposed to user, never changes)
	#     e: an edge id (as exposed to user, never changes)
	# rv,re: a 'raw' id (used by igraph, may change)

	# The purpose of these prefixes is to make errors easier to spot.
	# - methods on self._rg accept and return re and rv --- NEVER e and v.
	# - values that enter or leave this class are e and v --- NEVER re and rv.

	def __init__(self):
		self._rg = igraph.Graph(directed=self.is_directed())

		# force igraph to add the attributes
		self._rg.vs[self.V_ATTRIBUTE] = []
		self._rg.es[self.E_ATTRIBUTE] = []

		self._next_e = 0

		self._rv_map = {}
		self._v_map  = {}
		self._re_map = {}
		self._e_map  = {}

	def _v_from_rv(self, rv): return self._v_map[rv]
	def _rv_from_v(self, v):  return self._rv_map[v]
	def _e_from_re(self, re): return self._e_map[re]
	def _re_from_e(self, e):  return self._re_map[e]

	def has_vertex(self, v):
		return v in self._rv_map

	def add_vertices(self, vs):
		vs = list(vs)

		if len(vs) != len(set(vs)):
			raise ValueError('vertex {} specified multiple times'.format(repr(v)))

		for v in vs:
			if self.has_vertex(v):
				raise ValueError('Vertex {} already in graph!'.format(repr(v)))

		self._rg.add_vertices(len(vs))
		self._rg.vs[-len(vs):][self.V_ATTRIBUTE] = vs
		self._update_v_maps()

	def _update_v_maps(self):
		self._rv_map.clear()
		self._v_map.clear()
		for rv,v in enumerate(self._rg.vs[self.V_ATTRIBUTE]):
			self._rv_map[v] = rv
			self._v_map[rv] = v

	def _update_e_maps(self):
		self._re_map.clear()
		self._e_map.clear()
		for re,e in enumerate(self._rg.es[self.E_ATTRIBUTE]):
			self._re_map[e] = re
			self._e_map[re] = e

	def _predict_next_re(self):
		return self._rg.ecount()

	def add_edge(self, v1, v2, **kwargs):
		re = self._predict_next_re()

		e = self._next_e
		self._next_e += 1

		self._re_map[e] = re
		self._e_map[re] = e

		rv1, rv2 = self._rv_from_v(v1), self._rv_from_v(v2)
		self._rg.add_edge(rv1, rv2, **kwargs)
		self._rg.es[self._rg.ecount()-1][self.E_ATTRIBUTE] = e

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
		raise NotImplementedError()

	# General method for obtaining the other vertex of an edge...
	#  but ONLY if specified vertex is a valid target. (else, error)
	def edge_source_given_target(self, e, vTarget):
		raise NotImplementedError()

	def is_directed(self):
		raise NotImplementedError()

	def spanning_forest(self):
		''' Returns an arbitrary spanning tree forest as a dict of {v: edgeFromParent}

		The root of each tree is omitted from the dict. '''
		backedges = {}

		def handle_tree_edge(g,e,source):
			target = g.edge_target_given_source(e,source)
			backedges[target] = e

		self.bfs_full(tree_edge = handle_tree_edge)
		return backedges

	def cycle_basis(self):
		''' Generate and return a cycle basis for the graph as a list of paths.

		A cycle basis is a minimal-size set of cycles such that any cycle in the
		graph can be defined as the XOR sum of cycles in the basis.

		Note that, generally speaking, the cycle basis of a graph is NOT unique.'''

		# Algorithm:
		# - Make a spanning tree/forest. ANY tree will do.
		# - For each non-tree edge, traverse the tree between the endpoints to complete a cycle.

		eBackEdges = self.spanning_forest()

		# helper method - returns list of edges from root to v
		def path_from_root(v):
			result = []

			while v in eBackEdges:
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

class UndirectedGraph(BaseGraph):
	def _edge_other_endpoint_impl(self, e, v, name='an endpoint'):
		source,target = self.edge_endpoints(e)
		if   v == source:
			return target
		elif v == target:
			return source
		else:
			raise ValueError('vertex {} cannot be {} of edge'
				' {} (which connects {} to {})'.format(v,name,e,source,target))

	def edge_target_given_source(self, e, v):
		return self._edge_other_endpoint_impl(e, v, 'the source')

	def edge_source_given_target(self, e, v):
		return self._edge_other_endpoint_impl(e, v, 'the target')

	def is_directed(self):
		return False

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

g = UndirectedGraph()
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
