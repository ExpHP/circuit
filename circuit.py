
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

import filetypes.internal.graph as graphio
import graph.path as vpath
import graph.cyclebasis
import networkx as nx

import util

__all__ = [
	'CircuitBuilder',
	'MeshCurrentSolver',
	'compute_circuit_currents',
	'validate_circuit',
	'save_circuit',
	'load_circuit',
]

# attribute names used internally by circuits
EATTR_SOURCE = 'src' # defines sign of voltage; not necessarily same as the graph's internal 'source'
EATTR_RESISTANCE = 'resistance'
EATTR_VOLTAGE = 'voltage'

# attribute names when a circuit is saved
INTERNAL_TO_SAVED_EATTR = {
	EATTR_SOURCE:     'voltsrc',
	EATTR_RESISTANCE: 'res',
	EATTR_VOLTAGE:    'volt',
}
SAVED_TO_INTERNAL_EATTR = util.dict_inverse(INTERNAL_TO_SAVED_EATTR)

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

def save_circuit(circuit, path):
	g = map_edge_attributes(circuit, INTERNAL_TO_SAVED_EATTR)
	graphio.write_networkx(g, path)

def load_circuit(path):
	# FIXME remove this later
	if path.endswith('.gpickle'): # old format
		return nx.read_gpickle(path)
	else:
		g = graphio.read_networkx(path)
		circuit = map_edge_attributes(g, SAVED_TO_INTERNAL_EATTR)
		return circuit

def circuit_edge_sign(circuit, s, t):
	positive_source = circuit.edge[s][t][EATTR_SOURCE]
	assert positive_source in (s,t)
	return +1.0 if s == positive_source else -1.0

def circuit_path_voltage(circuit, path):
	acc = 0.0
	for s,t in vpath.edges(path):
		acc += circuit_edge_sign(circuit,s,t) * circuit.edge[s][t][EATTR_VOLTAGE]
	return acc

def map_edge_attributes(g, d):
	copy = copy_without_attributes(g)
	for old,new in d.items():
		attr = nx.get_edge_attributes(g, old)
		nx.set_edge_attributes(copy, new, attr)
	return copy

def copy_without_attributes(g):
	result = nx.Graph()
	result.add_nodes_from(g)
	result.add_edges_from(g.edges())
	return result

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

class cached_property:
	''' A member function decorator which provides a clean way to defer computation.

	A property-like object which is automatically backed by a member.
	A value may be assigned to it via ``=`` (like a property) or by explicitly calling
	 ``put(value)``.  To obtain its value, one must explicitly call ``get()``; if no
	 value has been stored, it will call the decorated function to compute a new value,
	 storing and returning it.

	Invoking ``invalidate()`` will reset the member (making it as though no value has
	 been set).  This allows one to arbitrarily defer an expensive computation by invoking
	 ``invalidate()`` now and only invoking ``get()`` once the value is needed (which will
	 then perform the computation).

	Example:

	>>> class Foo:
	...   @cached_property
	...   def x(self):
	...     print('computing!!!')
	...     return 10
	...
	>>> foo = Foo()
	>>> foo.x.get()        # prints 'computing!!!', => 10
	>>> foo.x.get()        # => 10
	>>>
	>>> foo.x = 3
	>>> foo.x.put(3)       # also valid
	>>> foo.x.get()        # => 3
	>>>
	>>> foo.x.invalidate()
	>>> foo.x.get()        # prints 'computing!!!', => 10
	'''
	def __init__(self, func):
		self.member   = self.__generate_member_name()
		self.sentinel = object()
		self.func     = func

	def __get__(self, obj, objtype=None):
		return bound_cached_property(self, obj)

	def __set__(self, obj, value):
		self.put(obj, value)

	def __generate_member_name(self):
		# FIXME I imagine this doesn't play well with serialization
		return '_cached__{}'.format(id(self))

	def get(self, obj):
		if getattr(obj, self.member, self.sentinel) is self.sentinel:
			setattr(obj, self.member, self.func(obj))
		assert getattr(obj, self.member) is not self.sentinel
		return getattr(obj, self.member)

	def put(self, obj, value):
		setattr(obj, self.member, value)

	def invalidate(self, obj):
		setattr(obj, self.member, self.sentinel)

class bound_cached_property:
	__doc__ = cached_property.__doc__

	def __init__(self, prop, obj):
		self.prop = prop
		self.obj = obj

	def get(self):        return self.prop.get(self.obj)
	def put(self, value): self.prop.put(self.obj, value)
	def invalidate(self): self.prop.invalidate(self.obj)

#------------------------------------------------------------
# Needless to say, computing the currents of a circuit could very easily be provided via a
#  standalone function. (In earlier versions, it WAS). However, some of the intermediate
#  data structures are very time consuming to obtain (such as a good-quality cycle basis).
#
# MeshCurrentSolver exists in order to keep them around, allowing new results to be derived
#  from old results when a modification is made to the graph.
#
# Its only responsibility is knowing what happens to each data structure for a given
#  modification to the graph.
#
#   * Things which can be updated in place: Call the appropriate method.
#
#   * Things which are still valid: Leave them alone.
#
#   * Things which must be recomputed from scratch:  Defer their computation until is
#     known they will be needed (via use of cached_property)
#
# The class itself contains NO logic for computation.
#
class MeshCurrentSolver:
	'''
	Computes currents in a circuit via mesh current analysis.

	Computes the currents in a circuit, and provides efficient
	methods for computing new currents in response to various
	modifications to the graph.
	'''
	def __init__(self, circuit, cyclebasis, cbupdater):
		validate_circuit(circuit)

		self.g = circuit
		self.cbupdater = cbupdater

		self.cbupdater.init(cyclebasis)

		# Invalidate everything
		self.cyclebasis.invalidate()
		self.cycles_from_edge.invalidate()
		self.voltage_vector.invalidate()
		self.resistance_matrix.invalidate()
		self.cycle_currents.invalidate()

	def delete_node(self, v):
		'''
		Removes a vertex and all associated edges from the circuit.
		'''
		# update in-place
		self.g.remove_node(v)
		self.cbupdater.remove_vertex(self.g, v)

		self.cyclebasis.invalidate()
		self.cycles_from_edge.invalidate()
		self.voltage_vector.invalidate()
		self.resistance_matrix.invalidate()
		self.cycle_currents.invalidate()

	def multiply_nearby_resistances(self, v, factor):
		'''
		Multiplies the resistance of edges connected to a vertex by a scalar factor.
		'''
		for t in self.g.neighbors(v):
			self.g.edge[v][t][EATTR_RESISTANCE] *= factor

		self.cyclebasis       # still valid!
		self.cycles_from_edge # still valid!
		self.voltage_vector   # still valid!
		self.resistance_matrix.invalidate()
		self.cycle_currents.invalidate()

	def assign_nearby_resistances(self, v, value):
		'''
		Assigns a value to the resistance of all edges connected to a vertex.
		'''
		for t in self.g.neighbors(v):
			self.g.edge[v][t][EATTR_RESISTANCE] = value

		self.cyclebasis       # still valid!
		self.cycles_from_edge # still valid!
		self.voltage_vector   # still valid!
		self.resistance_matrix.invalidate()
		self.cycle_currents.invalidate()

	# FIXME the cached properties really aren't meant to be part of the public api
	@cached_property
	def cyclebasis(self):
		cb = self.cbupdater.get_cyclebasis()

		# NOTE: this test is here because it is one of the only few paths that code
		#  reliably passes through where the current state of the modified cyclebasis
		#  and graph are both available.

		# FIXME: whatever happened to validate_cyclebasis?
		if len(cb) != len(nx.cycle_basis(self.g)):

			# FIXME: This is an error (rather than assertion) due to an unresolved issue
			#  with the builder updater algorithm;  I CANNOT say with confidence that
			#  this will not occur. -_-
			raise RuntimeError('Cyclebasis has incorrect rank ({}, need {}).'.format(
				len(cb),len(nx.cycle_basis(self.g))))

		return cb

	@cached_property
	def cycles_from_edge(self):
		return compute_cycles_from_edge(self.g, self.cyclebasis.get())

	@cached_property
	def voltage_vector(self):
		return compute_voltage_vector(self.g, self.cyclebasis.get())

	@cached_property
	def resistance_matrix(self):
		return compute_resistance_matrix(self.g, self.cyclebasis.get(), self.cycles_from_edge.get())

	@cached_property
	def cycle_currents(self):
		return compute_cycle_currents(self.resistance_matrix.get(), self.voltage_vector.get(), self.cyclebasis.get())

	def get_all_currents(self):
		'''
		Compute all currents in the circuit.

		The return value is a dict ``d`` such that ``d[s,t]`` (for an existing edge ``(s,t)``)
		is the (signed) current that flows from ``s`` to ``t``. It is guaranteed that
		``d[s,t] == -(d[t,s])``.
		'''
		# NOTE: of course, the guarantee that d[s,t] == -d[t,s] is written under the assumption
		#       that d[s,t] is not NaN
		d = compute_all_edge_currents(self.g, self.cycle_currents.get(), self.cycles_from_edge.get())
		assert all(d[t,s] == -d[s,t] for s,t in d)
		return d

	def get_current(self, s, t):
		'''
		Compute signed current current flowing from ``s`` to ``t``.

		It is a ``KeyError`` if no such edge exists in the graph.
		'''
		if not self.g.has_edge(s,t):
			raise KeyError('no such edge: {}'.format(repr((s,t))))
		return compute_single_edge_current(self.g, self.cycle_currents.get(), self.cycles_from_edge.get(), s, t)

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
	# special case for no cycles (which otherwise makes a singular matrix)
	if len(cyclebasis) == 0:
		return np.array([], dtype=v_vec.dtype)

	solver = spla.factorized(r_mat.tocsc())
	return solver(v_vec).reshape([len(cyclebasis)])

def compute_single_edge_current(g, cycle_currents, cycles_from_edge, s, t):
	ecycles = util.edictget(cycles_from_edge, (s,t))
	esign   = circuit_edge_sign(g, s, t)
	return esign * sum(cycle_currents[i] * csign for i, csign in ecycles)

def compute_all_edge_currents(g, cycle_currents, cycles_from_edge):
	result = {}
	for s,t in g.edges():
		result[s,t] = compute_single_edge_current(g, cycle_currents, cycles_from_edge, s, t)

	result.update({(t,s):-value for (s,t),value in result.items()})
	return result

#------------------------------------------------------------

# more ergonomic than MeshCurrentSolver when there's no need to update the graph
def compute_circuit_currents(circuit, cyclebasis=None):
	'''
	Computes an edge attribute dictionary of edge -> current for a circuit.

	The return value is a dict ``d`` such that ``d[s,t]`` (for an existing edge ``(s,t)``)
	is the (signed) current that flows from ``s`` to ``t``. It is guaranteed that
	``d[s,t] == -(d[t,s])``.

	If no cyclebasis is provided, one will be automatically generated.  For large graphs,
	however, the computation time can be significantly reduced by providing a custom
	cyclebasis with a small total edge count.
	'''
	from components import cyclebasis_provider
	if cyclebasis is None:
		cyclebasis = cyclebasis_provider.last_resort().new_cyclebasis(circuit)

	solver = MeshCurrentSolver(circuit, cyclebasis, cyclebasis_provider.dummy_cbupdater())

	d = solver.get_all_currents()
	assert all(d[t,s] == -d[s,t] for s,t in d)
	return d

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
	currents = compute_circuit_currents(circuit)

	for s,t in ('ab', 'bc', 'ca', 'xy', 'yz', 'zx'):
		assertNear(currents[s,t], +2.5)
		assertNear(currents[t,s], -2.5)

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
	currents = compute_circuit_currents(circuit)

	assertNear(currents['Dn','Lt'], +5.0)
	assertNear(currents['Dn','Rt'], -1.0)
	assertNear(currents['Up','Dn'], +4.0)
	assertNear(currents['Up','Lt'], -5.0)
	assertNear(currents['Up','Rt'], +1.0)

def test_get_current_consistency():
	import random
	from components import cyclebasis_provider
	g = nx.gnm_random_graph(5,16)
	builder = CircuitBuilder(g)
	for s,t in g.edges():
		builder.make_component(s, t, resistance=random.random(), voltage=random.random())
	circuit = builder.build()

	# freestanding function results
	currents = compute_circuit_currents(circuit)

	# class results
	cbprovider = cyclebasis_provider.last_resort()
	solver = MeshCurrentSolver(circuit, cbprovider.new_cyclebasis(circuit), cbprovider.cbupdater())

	# consistent?
	for s,t in g.edges():
		assertNear(solver.get_current(s,t), currents[s,t])
		assertNear(solver.get_current(t,s), currents[t,s])

def assertNear(a,b,eps=1e-7):
	assert abs(a-b) < eps

def iterN(n, f, *a, **kw):
	for _ in range(n):
		f(*a, **kw)

test_two_separate_loops()
test_two_loop_circuit()
test_get_current_consistency()

