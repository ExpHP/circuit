
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

import graph.path as vpath
import graph.cyclebasis
import networkx as nx

from resistances_common import *

import util

__all__ = [
	'CircuitBuilder',
	'MeshCurrentSolver',
	'validate_circuit',
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
		assert validate_circuit(copy)
		return copy

# Function which more or less defines a `circuit`, as I'd rather not make a class.
# returns True or throws an exception (the True return value is to permit its use
#  in an assert statement)
def validate_circuit(circuit):
	# Check graph type...
	if circuit.is_directed() or circuit.is_multigraph():
		raise ValueError('Circuit must be an undirected non-multigraph (nx.Graph())')

	# Check for missing attributes...
	eattrs = (EATTR_VOLTAGE, EATTR_RESISTANCE, EATTR_SOURCE)
	for name in eattrs:
		d = nx.get_edge_attributes(circuit, name)
		d.update({(t,s):val for ((s,t),val) in d.items()}) # have both directions for easy lookup

		if any(e not in d for e in circuit.edges()):
			raise ValueError('There are missing edge attributes.  All edges must define: {}'.format(expected_eattrs))

	# Check for illegal attribute contents...
	# The value of EATTR_SOURCE must be one of the edge's endpoints
	if any(circuit.edge[s][t][EATTR_SOURCE] not in (s,t) for (s,t) in circuit.edges()):
		raise ValiueError('An edge has an invalid "{}" attribute (it must equal one of the edge\'s endpoints)')

	return True

def circuit_edge_sign(circuit, s, t):
	positive_source = circuit.edge[s][t][EATTR_SOURCE]
	assert positive_source in (s,t)
	return +1.0 if s == positive_source else -1.0

def circuit_path_voltage(circuit, path):
	acc = 0.0
	for s,t in vpath.edges(path):
		acc += circuit_edge_sign(circuit,s,t) * circuit.edge[s][t][EATTR_VOLTAGE]
	return acc

#------------------------------------------------------------

# @provides(member_name):
# Member function decorator for a function which caches its results in a member of
#  the class.

# To use properly, set the member to None to signal when it has become invalidated.
# Whenever the stored value is needed, call the decorated function instead of accessing
#  the member directly.

# TODO: Look into better, more idiomatic solutions to the problem of cache invalidation;
# This will clearly become unmaintainable very quickly.
class provides:
	def __init__(self, member):
		self.member = member

	def __call__(self, func):

		def wrapped(obj, *a, **kw):
			if getattr(obj, self.member) is None:
				setattr(obj, self.member, func(obj, *a, **kw))

			assert getattr(obj, self.member) is not None
			return getattr(obj, self.member)

		return wrapped


class MeshCurrentSolver:

	def __init__(self, circuit, cyclebasis=None, is_planar=False):
		validate_circuit(circuit)

		self._g = circuit
		self._is_planar = is_planar

		# Invalidate everything
		self._cycle_basis      = cyclebasis
		self._cycles_from_edge = None
		self._cycle_currents   = None

	def delete_node(self, v):
		self._g.remove_node(v)

		# Update what we can
		if self._is_planar and self._cycle_basis is not None:
			self._cycle_basis      = graph.cyclebasis.planar.without_vertex(self._cycle_basis, v)
			self._cycles_from_edge = None
			self._cycle_currents   = None
		else:
			self._cycle_basis      = None
			self._cycles_from_edge = None
			self._cycle_currents   = None

		assert len(self._cycle_basis) == len(nx.cycle_basis(self._g))

	def multiply_nearby_resistances(self, v, factor):
		for t in self._g.neighbors(v):
			self._g.edge[v][t][EATTR_RESISTANCE] *= factor

		self._cycle_basis      # still valid!
		self._cycles_from_edge # still valid!
		self._cycle_currents   = None

	def assign_nearby_resistances(self, v, value):
		for t in self._g.neighbors(v):
			self._g.edge[v][t][EATTR_RESISTANCE] = value

		self._cycle_basis      # still valid!
		self._cycles_from_edge # still valid!
		self._cycle_currents   = None

	@provides('_cycle_basis')
	def _acquire_cycle_basis(self):
		if self._is_planar:
			return compute_planar_cycle_basis(self._g)
		else:
			return compute_default_cycle_basis(self._g)
		assert False

	@provides('_cycles_from_edge')
	def _acquire_cycles_from_edge(self):
		g = self._g
		cyclebasis = self._acquire_cycle_basis()

		return compute_cycles_from_edge(g, cyclebasis)

	@provides('_cycle_currents')
	def _acquire_cycle_currents(self):
		g = self._g
		cyclebasis = self._acquire_cycle_basis()
		cycles_from_edge = self._acquire_cycles_from_edge()

		V = compute_voltage_vector(g, cyclebasis)
		R = compute_resistance_matrix(g, cyclebasis, cycles_from_edge)

		return compute_cycle_currents(R, V, cyclebasis)

	def get_current(self, s, t):
		g = self._g
		cycles_from_edge = self._acquire_cycles_from_edge()
		cycle_currents = self._acquire_cycle_currents()

		ecycles = util.edictget(cycles_from_edge, (s,t))
		return circuit_edge_sign(g,s,t) * sum(cycle_currents[cycleId] * sign for cycleId, sign in ecycles)

#------------------------------------------------------------

# Don't make these member functions;
# They are laid out below as free functions to explicitly spell out all of the
#  data dependencies

def compute_planar_cycle_basis(g):
	xs,ys = nx.get_node_attributes(g, VATTR_X), nx.get_node_attributes(g, VATTR_Y)
	return planar_cycle_basis.planar_cycle_basis_nx(g, xs, ys)

def compute_default_cycle_basis(g):
	cyclebasis = nx.cycle_basis(g)
	for cycle in cyclebasis:
		cycle.append(cycle[0])
	return cyclebasis

def compute_cycles_from_edge(g, cyclebasis):
	cycles_from_edge = {e:[] for e in g.edges()}

	for pathI, path in enumerate(cyclebasis):
		for e in vpath.edges(path):
			sign = circuit_edge_sign(g, *e)

			ecycles = util.edictget(cycles_from_edge, e)
			ecycles.append((pathI, sign))
	return cycles_from_edge

def compute_voltage_vector(g, cyclebasis):
	return np.array([circuit_path_voltage(g, path) for path in cyclebasis])

def compute_resistance_matrix(g, cyclebasis, cycles_from_edge):
	# Build components of resistance matrix, in coo (COOrdinate format) sparse format
	R_vals = []
	R_rows = []
	R_cols = []
	for e in g.edges():
		s,t = e
		resistance = g.edge[s][t][EATTR_RESISTANCE]

		ecycles = util.edictget(cycles_from_edge, e)

		# generate terms corresponding to this edge, which are +r between cycles that cross the
		#  edge in the same direction, and -r between cycles that cross in opposite directions
		for (row, row_sign) in ecycles:
			R_rows.extend([row]*len(ecycles))
			R_cols.extend([col for (col,_) in ecycles])
			R_vals.extend([row_sign * col_sign * resistance for (_,col_sign) in ecycles])
			assert len(R_vals) == len(R_rows) == len(R_cols)

	return sparse.coo_matrix((R_vals, (R_rows, R_cols)), shape=(len(cyclebasis),)*2)

def compute_cycle_currents(r_mat, v_vec, cyclebasis):
	solver = spla.factorized(r_mat.tocsc())
	return solver(v_vec).reshape([len(cyclebasis)])

#------------------------------------------------------------

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
	solver = MeshCurrentSolver(circuit, is_planar=False)

	for s,t in ('ab', 'bc', 'ca', 'xy', 'yz', 'zx'):
		assertNear(solver.get_current(s,t), +2.5)

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
	solver = MeshCurrentSolver(circuit, is_planar=False)

	assertNear(solver.get_current('Dn','Lt'), +5.0)
	assertNear(solver.get_current('Dn','Rt'), -1.0)
	assertNear(solver.get_current('Up','Dn'), +4.0)
	assertNear(solver.get_current('Up','Lt'), -5.0)
	assertNear(solver.get_current('Up','Rt'), +1.0)

def assertNear(a,b,eps=1e-7):
	assert abs(a-b) < eps

def iterN(n, f, *a, **kw):
	for _ in range(n):
		f(*a, **kw)

test_two_separate_loops()
test_two_loop_circuit()

