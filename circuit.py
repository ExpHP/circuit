
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

import graphs.vertexpath as vpath
import networkx as nx

import util

__all__ = [
	'Circuit',
]

# produce a sign factor (+/- 1) based on which side we're traveling an
#  edge from.  This is to allow circuit components to appropriately
#  account for direction even though the graph is undirected.
def _sign_from_source_vertex(endpoints, source):
	s,t = endpoints
	if source == s: return  1.0
	if source == t: return -1.0
	raise ValueError('source not in endpoints')

# TODO: this ought not to be necessary; a path should be a class object
#       that is capable of producing this information on its own.
# (this hackish workaround won't even work in some cases; the edge list
#  data structure does not allow one to identify the (sole) vertex of a
#  0 edge path, or to identify the direction of a 1 edge path in an
#  undirected graph)
# (it also can't determine the correct order of a 2-cycle in an undirected
#  graph, but I'll let that slide)
def _vertices_from_path(g, edgelist):
	if len(edgelist) < 2: raise RuntimeError('This edge list is too short '
		'for its vertices to properly be determined.  Tell the developer '
		'to stop being lazy and implement a proper Path class already!')

	s1,t1 = g.edge_endpoints(edgelist[0])
	s2,t2 = g.edge_endpoints(edgelist[1])

	# HACK: Identify the SECOND vertex as the one shared by the first two edges;
	#  the other vertex must be the first one in the path.
	if   t1 in (s2,t2): first = s1
	elif s1 in (s2,t2): first = t1
	else: raise RuntimeError('Path invalid (neither vertex of first edge is in second edge)')

	vs = [first]
	for edge in edgelist:
		vs.append(g.edge_target_given_source(edge, vs[-1]))

	assert len(vs) == len(edgelist) + 1

	return vs

# the primary purpose of _vertices_from_path (yield (edge, source) pairs so that we
#  can determine the direction of edges as we iterate over them)
def _iter_path_with_sources(g, edgelist):
	vs = _vertices_from_path(g, edgelist)
	return iter(zip(edgelist, vs[:-1]))

# An electrical component which rests on a graph edge.
class Component:
	# Create a component given a (source, target) tuple specifying its positive direction.
	def __init__(self, endpoints, *, voltage=0.0, resistance=0.0):
		self._endpoints = endpoints
		self._potential = voltage
		self._resistance = resistance

	@classmethod
	def make_battery(cls, endpoints, voltage):
		obj = cls()
		obj._potential = voltage
		return obj

	@classmethod
	def make_resistor(cls, endpoints, resistance):
		obj = cls()
		obj._resistance = resistance
		return obj

	def direction_sign(self, sourceVertex):
		return _sign_from_source_vertex(self._endpoints, sourceVertex)

	def resistance(self):
		return self._resistance

	def voltage(self, sourceVertex):
		return self.direction_sign(sourceVertex) * self._potential


class Circuit(nx.Graph):
	def add_resistor(self, v1, v2, resistance):
		self.add_component(v1,v2, resistance=resistance, voltage=0.0)

	def add_battery(self, v1, v2, voltage):
		self.add_component(v1,v2, voltage=voltage, resistance=0.0)

	def add_component(self, v1, v2, *, voltage, resistance):
		component = Component((v1,v2), voltage=voltage, resistance=resistance)
		self.add_edge(v1,v2)
		self.edge[v1][v2]['_component'] = component

	def edge_component(self, e):
		s,t = e
		return self.edge[s][t]['_component']

	def path_total_voltage(self, path):
		acc = 0.0
		for source,target in vpath.edges(path):
			acc += self.edge_component((source,target)).voltage(source)
		return acc

	def compute_currents(self, cyclebasis=None):

		# Currents are computed using mesh current analysis;
		# We only compute a current for each cycle in the cycle basis.

		if cyclebasis is None:
			cyclebasis = self._default_cycle_basis()

		cycles_from_edge = self._generate_cycles_from_edge(cyclebasis)

		# Generate matrices
		V = self._generate_voltage_vector(cyclebasis)
		R = self._generate_resistance_matrix(cyclebasis, cycles_from_edge)

		# Solve linear system
		cycle_currents = _solve_sparse(R,V).reshape([len(cyclebasis)])

		edge_currents = self._compute_edge_currents(cycle_currents, cycles_from_edge)
		return edge_currents

	def _default_cycle_basis(self):
		cyclebasis = nx.cycle_basis(self)
		for cycle in cyclebasis:
			cycle.append(cycle[0])
		return cyclebasis

	# For each edge, generate a list of (index, sign) for each cycle that crosses it.
	# This is used to go back and forth between the cycle currents and the individual
	#  edge currents.
	def _generate_cycles_from_edge(self, cyclebasis):
		cycles_from_edge = {e:[] for e in self.edges()}

		for pathI, path in enumerate(cyclebasis):
			for e in vpath.edges(path):
				source,target = e
				sign = self.edge_component(e).direction_sign(source)

				ecycles = util.edictget(cycles_from_edge, e)
				ecycles.append((pathI, sign))
		return cycles_from_edge

	def _generate_voltage_vector(self, cyclebasis):
		return np.array([self.path_total_voltage(path) for path in cyclebasis])

	def _generate_resistance_matrix(self, cyclebasis, cycles_from_edge):
		# Build components of resistance matrix, in coo (COOrdinate format) sparse format
		R_vals = []
		R_rows = []
		R_cols = []
		for e in self.edges():
			component = self.edge_component(e)
			resistance = component.resistance()

			ecycles = util.edictget(cycles_from_edge, e)

			# generate terms corresponding to this edge, which are +r between cycles that cross the
			#  edge in the same direction, and -r between cycles that cross in opposite directions
			for (row, row_sign) in ecycles:
				R_rows.extend([row]*len(ecycles))
				R_cols.extend([col for (col,_) in ecycles])
				R_vals.extend([row_sign * col_sign * resistance for (_,col_sign) in ecycles])
				assert len(R_vals) == len(R_rows) == len(R_cols)

		return sparse.coo_matrix((R_vals, (R_rows, R_cols)), shape=(len(cyclebasis),)*2)

	def _compute_edge_currents(self, cycle_currents, cycles_from_edge):
		edge_currents = {}
		for e in self.edges():
			ecycles = util.edictget(cycles_from_edge, e)
			edge_currents[e] = sum(cycle_currents[cycleId] * sign for cycleId, sign in ecycles)

		return edge_currents

def _solve_sparse(mat,vec):
	solver = spla.factorized(mat.tocsc())
	return solver(vec)

def test_two_separate_loops():
	# A circuit with two connected components.
	circuit = Circuit()
	circuit.add_nodes_from(['a1', 'a2', 'a3', 'b1', 'b2', 'b3'])
	circuit.add_battery('a1', 'a2', 5.0)
	circuit.add_resistor('a2', 'a3', 1.0)
	circuit.add_resistor('a3', 'a1', 1.0)
	circuit.add_battery('b1', 'b2', 5.0)
	circuit.add_resistor('b2', 'b3', 1.0)
	circuit.add_resistor('b3', 'b1', 1.0)

	current = circuit.compute_currents()
	for v in current.values():
		assert(abs(v - 2.5) < 1E-10)

def test_two_loop_circuit():
	# {0: 5.0, 1: -0.99999999999999956, 2: 4.0, 3: -5.0, 4: 0.99999999999999956}
	circuit = Circuit()
	circuit.add_nodes_from(['top', 'bot', 'left', 'right'])
	circuit.add_battery('bot', 'left',  28.0)
	circuit.add_battery('bot', 'right',  7.0)
	circuit.add_resistor('top', 'bot',   2.0)
	circuit.add_resistor('top', 'left',  4.0)
	circuit.add_resistor('top', 'right', 1.0)
	currents = circuit.compute_currents()

	assertNear(util.edictget(currents, ('bot','left')),  +5.0)
	assertNear(util.edictget(currents, ('bot','right')), -1.0)
	assertNear(util.edictget(currents, ('top','bot')),   +4.0)
	assertNear(util.edictget(currents, ('top','left')),  -5.0)
	assertNear(util.edictget(currents, ('top','right')), +1.0)

def assertNear(a,b,eps=1e-7):
	assert abs(a-b) < eps

def iterN(n, f, *a, **kw):
	for _ in range(n):
		f(*a, **kw)

test_two_separate_loops()
test_two_loop_circuit()

