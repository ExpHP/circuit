
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

def sane_equals(a, b):
	''' An alternative to ``==`` which pounds numpy into freaking submission die die die '''
	if isinstance(a, np.ndarray) != isinstance(b, np.ndarray):
		return False
	elif isinstance(a, np.ndarray):
		return (a == b).all() # numpy: polymorphism wats that
	else:
		return a == b

#------------------------------------------------------------


# Sentinels
RECOMPUTE = object()
KEEP = object()

# NOTE: I still think this is a terrible idea, I just don't have any better idea
#       on how to defer (potentially useless) computations until they're needed
#       which doesn't drown the code in boilerplate.
class cached_property:
	''' A member function decorator which provides a clean way to defer computation.

	A property which is automatically backed by a member, with some slightly unusual
	 semantics for controlling how it is updated.

	The first time the member is accessed (if it hasn't been assigned a value), it
	 will call the decorated function, store and return the result.
	In most cases, assigning a value to the property will simply store that value.
	But there two special values:

	 * Assigning ``RECOMPUTE`` resets it. (next getter will call the function again)
	 * Assigning ``KEEP`` is a no-op. (just there for the reader's sake)

	Example:

	>>> class Foo:
	...   @cached_property()
	...   def x(self):
	...     print('computing!!!')
	...     return 10
	...
	>>>
	>>> foo = Foo()
	>>> foo.x             # => 10, prints 'computing!!!'
	>>> foo.x             # => 10
	>>> foo.x = 3
	>>> foo.x             # => 3
	>>> foo.x = KEEP
	>>> foo.x             # => 3
	>>> foo.x = RECOMPUTE
	>>> foo.x             # => 10, prints 'computing!!!'
	'''
	def __init__(self, recompute=RECOMPUTE, keep=KEEP):
		self.member = self.__generate_member_name()
		self.recompute = recompute
		self.keep = keep

	def __generate_member_name(self):
		return '_cached__{}'.format(id(self))

	def __call__(self, func):

		def getmbr(obj):      return getattr(obj, self.member)
		def setmbr(obj, val): setattr(obj, self.member, val)

		def getter(obj):
			if sane_equals(getmbr(obj), self.recompute):
				setmbr(obj, func(obj))
			assert not sane_equals(getmbr(obj), self.recompute)
			assert not sane_equals(getmbr(obj), self.keep)
			return getmbr(obj)

		def setter(obj, value):
			if not sane_equals(value, self.keep):
				setmbr(obj, value)

		return property(getter, setter)


class MeshCurrentSolver:

	def __init__(self, circuit, cyclebasis, cbupdater):
		validate_circuit(circuit)

		self.g = circuit
		self.cbupdater = cbupdater

		self.cbupdater.init(cyclebasis)

		# Invalidate everything
		self.cyclebasis       = RECOMPUTE
		self.cycles_from_edge = RECOMPUTE
		self.cycle_currents   = RECOMPUTE

	def delete_node(self, v):
		self.g.remove_node(v)
		self.cbupdater.remove_vertex(self.g, v)

		self.cyclebasis       = RECOMPUTE
		self.cycles_from_edge = RECOMPUTE
		self.cycle_currents   = RECOMPUTE

	def multiply_nearby_resistances(self, v, factor):
		for t in self.g.neighbors(v):
			self.g.edge[v][t][EATTR_RESISTANCE] *= factor

		self.cyclebasis       = KEEP
		self.cycles_from_edge = KEEP
		self.cycle_currents   = RECOMPUTE

	def assign_nearby_resistances(self, v, value):
		for t in self.g.neighbors(v):
			self.g.edge[v][t][EATTR_RESISTANCE] = value

		self.cyclebasis       = KEEP
		self.cycles_from_edge = KEEP
		self.cycle_currents   = RECOMPUTE

	@cached_property()
	def cyclebasis(self):
		return self.cbupdater.get_cyclebasis()

	@cached_property()
	def cycles_from_edge(self):
		return compute_cycles_from_edge(self.g, self.cyclebasis)

	@cached_property()
	def cycle_currents(self):
		V = compute_voltage_vector(self.g, self.cyclebasis)
		R = compute_resistance_matrix(self.g, self.cyclebasis, self.cycles_from_edge)

		return compute_cycle_currents(R, V, self.cyclebasis)

	def get_current(self, s, t):
		ecycles = util.edictget(self.cycles_from_edge, (s,t))
		esign   = circuit_edge_sign(self.g, s, t)
		return esign * sum(self.cycle_currents[i] * csign for i, csign in ecycles)

#------------------------------------------------------------

# Functions which actually compute stuff for MeshCurrentSolver.

# These are laid out as free functions so that all data dependencies are clearly
#  spelled out in the arguments.

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
	import cyclebasis_provider
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
	cbprovider = cyclebasis_provider.last_resort()
	solver = MeshCurrentSolver(circuit, cbprovider.new_cyclebasis(g), cbprovider.cbupdater())

	for s,t in ('ab', 'bc', 'ca', 'xy', 'yz', 'zx'):
		assertNear(solver.get_current(s,t), +2.5)

def test_two_loop_circuit():
	import cyclebasis_provider
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
	cbprovider = cyclebasis_provider.last_resort()
	solver = MeshCurrentSolver(circuit, cbprovider.new_cyclebasis(g), cbprovider.cbupdater())

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

