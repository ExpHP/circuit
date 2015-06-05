
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

import graphs.vertexpath as vpath
import networkx as nx

from resistances_common import *

import util

__all__ = [
	'Circuit',
	'CircuitBuilder',
]

# produce a sign factor (+/- 1) based on which side we're traveling an
#  edge from.  This is to allow circuit components to appropriately
#  account for direction even though the graph is undirected.
def _sign_from_source_vertex(endpoints, source):
	s,t = endpoints
	if source == s: return  1.0
	if source == t: return -1.0
	raise ValueError('source not in endpoints')

# For producing hardcoded circuits.
class CircuitBuilder():
	_DEFAULT_VOLTAGE    = 0.0
	_DEFAULT_RESISTANCE = 0.0

	def __init__(self, g):
		self._g = _copy_graph_without_attributes(g)

	# adds an edge or turns an existing edge into a battery (has a potential
	#  difference of +voltage from s to t, and 0 resistance)
	def make_battery(self, s, t, voltage):
		self.make_component(s, t, voltage=voltage)

	def make_resistor(self, s, t, resistance):
		self.make_component(s, t, resistance=resistance)

	# makes any sort of circuit component
	def make_component(self, s, t, *, resistance=0.0, voltage=0.0):
		g = self._g
		if not g.has_edge(s,t):
			g.add_edge(s,t)
		g.edge[s][t]['_voltage']    = voltage
		g.edge[s][t]['_resistance'] = resistance
		g.edge[s][t]['_source']     = s

	def build(self):
		g = self._g
		voltage    = nx.get_edge_attributes(g, '_voltage')
		resistance = nx.get_edge_attributes(g, '_resistance')
		sources    = nx.get_edge_attributes(g, '_source')

		assert set(voltage) == set(resistance) == set(sources)

		missing_edges = [e for e in g.edges() if (e not in voltage) and (e[::-1] not in voltage)]
		for e in missing_edges:
			voltage[e]     = self._DEFAULT_VOLTAGE
			resistances[e] = self._DEFAULT_RESISTANCE
			sources[e]     = e[0]

		copy = _copy_graph_without_attributes(g)
		nx.set_edge_attributes(copy, EATTR_VOLTAGE, voltage)
		nx.set_edge_attributes(copy, EATTR_RESISTANCE, resistance)
		nx.set_edge_attributes(copy, EATTR_SOURCE, sources)
		return Circuit(copy)

class Circuit:

	def __init__(self, g):
		if g.is_directed():
			raise ValueError('Directed graphs not supported.')
		if g.is_multigraph():
			raise ValueError('Multigraphs not supported.')
		self._g = g

	def edge_sign(self, s, t):
		positive_source = self._g.edge[s][t][EATTR_SOURCE]

		assert positive_source in (s,t)
		return +1.0 if s == positive_source else -1.0

	def path_total_voltage(self, path):
		acc = 0.0
		for s,t in vpath.edges(path):
			acc += self.edge_sign(s,t) * self._g.edge[s][t][EATTR_VOLTAGE]
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
		cyclebasis = nx.cycle_basis(self._g)
		for cycle in cyclebasis:
			cycle.append(cycle[0])
		return cyclebasis

	# For each edge, generate a list of (index, sign) for each cycle that crosses it.
	# This is used to go back and forth between the cycle currents and the individual
	#  edge currents.
	def _generate_cycles_from_edge(self, cyclebasis):
		cycles_from_edge = {e:[] for e in self._g.edges()}

		for pathI, path in enumerate(cyclebasis):
			for e in vpath.edges(path):
				sign = self.edge_sign(*e)

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
		for e in self._g.edges():
			s,t = e
			resistance = self._g.edge[s][t][EATTR_RESISTANCE]

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
		for e in self._g.edges():
			ecycles = util.edictget(cycles_from_edge, e)
			edge_currents[e] = sum(cycle_currents[cycleId] * sign for cycleId, sign in ecycles)

		return edge_currents

def _solve_sparse(mat,vec):
	solver = spla.factorized(mat.tocsc())
	return solver(vec)

def _copy_graph_without_attributes(g):
	cls = type(g)
	result = cls()
	result.add_nodes_from(g)
	result.add_edges_from(g.edges())
	return result

def test_two_separate_loops():
	# A circuit with two connected components.
	g = nx.Graph()
	g.add_path('abca')
	g.add_path('xyzx')
	builder = CircuitBuilder(g)
	builder.make_battery('a', 'b', 5.0)
	builder.make_resistor('b', 'c', 1.0)
	builder.make_resistor('c', 'a', 1.0)
	builder.make_battery('x', 'y', 5.0)
	builder.make_resistor('y', 'z', 1.0)
	builder.make_resistor('z', 'x', 1.0)
	circuit = builder.build()

	currents = circuit.compute_currents()
	for s,t in ('ab', 'bc', 'ca', 'xy', 'yz', 'zx'):
		assertNear(util.edictget(currents, (s,t)), +2.5)

def test_two_loop_circuit():
	# {0: 5.0, 1: -0.99999999999999956, 2: 4.0, 3: -5.0, 4: 0.99999999999999956}
	g = nx.Graph()
	g.add_path(['Up','Rt','Dn','Lt','Up'])
	g.add_edge('Dn','Up')
	builder = CircuitBuilder(g)
	builder.make_battery('Dn', 'Lt', 28.0)
	builder.make_battery('Dn', 'Rt',  7.0)
	builder.make_resistor('Up', 'Dn', 2.0)
	builder.make_resistor('Up', 'Lt', 4.0)
	builder.make_resistor('Up', 'Rt', 1.0)
	circuit = builder.build()

	currents = circuit.compute_currents()

	assertNear(util.edictget(currents, ('Dn','Lt')), +5.0)
	assertNear(util.edictget(currents, ('Dn','Rt')), -1.0)
	assertNear(util.edictget(currents, ('Up','Dn')), +4.0)
	assertNear(util.edictget(currents, ('Up','Lt')), -5.0)
	assertNear(util.edictget(currents, ('Up','Rt')), +1.0)

def assertNear(a,b,eps=1e-7):
	assert abs(a-b) < eps

def iterN(n, f, *a, **kw):
	for _ in range(n):
		f(*a, **kw)

test_two_separate_loops()
test_two_loop_circuit()

