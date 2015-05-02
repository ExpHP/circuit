
import random

import igraph

from . import algorithm

# For methods which are not affected by directedness
# (or which are currently written in a directed-agnostic manner)
class AdjacencyListBase:

	def __init__(self):
		self._next_e = 0

		self._adj = {}  # dict of sets of edges
		self._edge_endpoints = {}
		self._edge_attributes = {}

	def num_vertices(self):
		return len(self._adj)

	def num_edges(self):
		return len(self._edge_endpoints)

	def vertices(self):
		return iter(self._adj)

	def edges(self):
		return iter(self._edge_endpoints)

	def edge_endpoints(self, e):
		return self._edge_endpoints[e]

	def edge_attribute(self, e, attr):
		return self._edge_attributes[e][attr]

	def has_vertex(self, v):
		return v in self._adj

	def random_vertex(self):
		# NOTE: making a list is, of course, O(V) in complexity :/
		# Also note that dict.popitem() is arbitrary; not stictly random.
		return random.choice(list(self._adj))

	def add_vertices(self, vs):
		vs = list(vs)

		if len(vs) != len(set(vs)):
			raise ValueError('vertex specified multiple times')

		if any(map(self.has_vertex,vs)):
			raise ValueError('Vertex {} already in graph!'.format(repr(v)))

		for v in vs:
			self._adj[v] = set()

	def delete_vertices(self, vs):
		es = set()
		for v in vs:
			es.update(self.incident_edges(v))

		self.delete_edges(es)
		for v in vs:
			del self._adj[v]

	def all_edges(self, v1, v2):
		result = []
		for e in self.out_edges(v1):
			if self.edge_target_given_source(e,v1) == v2:
				result.append(e)
		return result

	def arbitrary_edge(self, v1, v2):
		return self.all_edges(v1, v2)[0]

# Methods which must change to take directedness into account (keep this list small)
class UndirectedGraph(AdjacencyListBase, algorithm.AlgorithmMixin):

	def is_directed(self):
		return False

	def add_edge(self, v1, v2, **kwargs):
		e = self._next_e
		self._next_e += 1

		self._adj[v1].add(e)
		self._adj[v2].add(e)
		self._edge_endpoints[e] = (v1,v2)
		self._edge_attributes[e] = dict(kwargs)

	def delete_edges(self, es):
		for e in es:
			source, target = self.edge_endpoints(e)
			self._adj[source].remove(e)
			self._adj[target].remove(e)

			del self._edge_endpoints[e]
			del self._edge_attributes[e]

	def incident_edges(self, v):
		return iter(self._adj[v])
	def in_edges(self, v):
		return iter(self._adj[v])
	def out_edges(self, v):
		return iter(self._adj[v])

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


# Alternate graph storage type
class IgraphBaseGraph:

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

	def delete_vertices(self, vs):
		rvs = list(self._rv_from_v(v) for v in vs)
		self._rg.delete_vertices(rvs)
		self._update_v_maps()
		self._update_e_maps()

	def add_vertices(self, vs):
		vs = list(vs)

		if len(vs) != len(set(vs)):
			raise ValueError('vertex specified multiple times')

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
		eobj = self._igraph_edge(e)
		rvSource = eobj.source
		rvTarget = eobj.target
		return self._v_from_rv(rvSource), self._v_from_rv(rvTarget)

	def edge_attribute(self, e, attr):
		return self._igraph_edge(e)[attr]

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

class IgraphUndirectedGraph(IgraphBaseGraph, algorithm.AlgorithmMixin):
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
