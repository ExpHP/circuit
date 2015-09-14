
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

import networkx as nx

import defect.filetypes.internal as fileio
from defect.components import cyclebasis_provider
import defect.graph.path as vpath
from defect.util import dict_inverse, edictget

__all__ = [
	'CircuitBuilder',
	'MeshCurrentSolver',
	'compute_circuit_currents',
	'validate_circuit',
	'save_circuit',
	'load_circuit',
	'circuit_path_voltage',
	'circuit_path_resistance',
	'circuit_edge_sign',
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
SAVED_TO_INTERNAL_EATTR = dict_inverse(INTERNAL_TO_SAVED_EATTR)

# produce a sign factor (+/- 1) based on which side we're traveling an
#  edge from.  This is to allow circuit components to appropriately
#  account for direction even though the graph is undirected.
def _sign_from_source_vertex(endpoints, source):
	s,t = endpoints
	if source == s: return  1.0
	if source == t: return -1.0
	raise ValueError('source not in endpoints')

# For generating circuits.
class CircuitBuilder():
	'''
	Create a ``circuit``.

	There isn't really a ``circuit`` type, so ``CircuitBuilder`` provides the
	protocol for making one.  The result will be a ``networkx`` ``Graph`` with
	certain attributes set.

	>>> builder = CircuitBuilder()
	>>> builder.make_resistor('a', 'b', 1.0)
	>>> circuit = builder.build()
	>>> circuit.number_of_nodes()
	2
	'''
	# Fun fact: This module used to have a "Circuit" class.  I got rid of it because
	#   I was wasting too much time worrying about its API.

	_DEFAULT_VOLTAGE    = 0.0
	_DEFAULT_RESISTANCE = 0.0

	def __init__(self, g=None):
		if g is None:
			g = nx.Graph()
		self._g = _copy_graph_without_attributes(g)

	# class invariant:  Any edge on `self._g` which defines at least one edge property
	#                   (source, voltage, resistance) will define all three.
	# (however, an edge may have NO properties)

	def add_node(self, s):
		self._g.add_node(s)

	def make_battery(self, s, t, voltage):
		'''
		Create or turn the existing edge `(s,t)` into a voltage source.

		The edge will have zero resistance.
		The order of ``s`` and ``t`` are significant; ``voltage`` is interpreted
		as ``Vt - Vs``, where ``(Vs,Vt)`` are the potential at ``s`` and ``t``.
		'''
		self.make_component(s, t, voltage=voltage)

	def make_resistor(self, s, t, resistance):
		'''
		Create or turn the existing edge `(s,t)` into a resistor.

		The edge will have zero voltage.
		'''
		self.make_component(s, t, resistance=resistance)

	def make_component(self, s, t, *, resistance=0.0, voltage=0.0):
		'''
		Create or turn the existing edge `(s,t)` into an arbitrary component.

		The order of ``s`` and ``t`` are significant; ``voltage`` is interpreted
		as ``Vt - Vs``, where ``(Vs,Vt)`` are the potential at ``s`` and ``t``.
		'''
		g = self._g
		if not g.has_edge(s,t):
			g.add_edge(s,t)
		g.edge[s][t]['_voltage']    = voltage
		g.edge[s][t]['_resistance'] = resistance
		g.edge[s][t]['_source']     = s

	def build(self):
		'''
		Produce a ``circuit``.
		'''
		g = self._g
		voltages    = nx.get_edge_attributes(g, '_voltage')
		resistances = nx.get_edge_attributes(g, '_resistance')
		sources     = nx.get_edge_attributes(g, '_source')

		# class invariant of CircuitBuilder; no attribute ever appears without the other two
		assert set(voltages) == set(resistances) == set(sources)

		# this covers edges present in the initial graph (passed into the constructor)
		# which were not addressed via make_resistor and friends
		missing_edges = [e for e in g.edges() if (e not in voltages) and (e[::-1] not in voltages)]
		for e in missing_edges:
			voltages[e]    = self._DEFAULT_VOLTAGE
			resistances[e] = self._DEFAULT_RESISTANCE
			sources[e]     = e[0]

		copy = _copy_graph_without_attributes(g)
		nx.set_edge_attributes(copy, EATTR_VOLTAGE, voltages)
		nx.set_edge_attributes(copy, EATTR_RESISTANCE, resistances)
		nx.set_edge_attributes(copy, EATTR_SOURCE, sources)
		assert validate_circuit(copy)
		return copy

def validate_circuit(circuit):
	'''
	Checks that the input ``networkx`` Graph meets all the criteria of a ``circuit``.

	This exists because there is no actual ``circuit`` class.  Think of this as "checking
	the class invariants."
	'''
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
	'''
	Write a circuit to a file.
	'''
	g = map_edge_attributes(circuit, INTERNAL_TO_SAVED_EATTR)
	fileio.graph.write_networkx(g, path)

def load_circuit(path):
	'''
	Read a circuit from a file.
	'''
	# FIXME remove this later
	if path.endswith('.gpickle'): # old format
		return nx.read_gpickle(path)
	else:
		g = fileio.graph.read_networkx(path)
		circuit = map_edge_attributes(g, SAVED_TO_INTERNAL_EATTR)
		return circuit

def circuit_edge_sign(circuit, s, t):
	'''
	Get the sign of a directed edge, relative to voltage.

	Each edge in a circuit has a fixed "source" vertex which is used
	to define the sign of its voltage.  This returns `+1.0` if `s`
	and `t` are in the correct order, and `-1.0` if they are reversed.
	'''
	positive_source = circuit.edge[s][t][EATTR_SOURCE]
	assert positive_source in (s,t)
	return +1.0 if s == positive_source else -1.0

def circuit_path_voltage(circuit, path):
	'''
	Get the total EMF produced by voltage sources along a path.

	The path voltage around a cycle may be nonzero, as it only takes fixed
	voltage sources into account (it does not account for potential
	differences caused by resistance).

	>>> import networkx as nx
	>>> builder = CircuitBuilder(nx.cycle_graph(5))
	>>> builder.make_battery(2, 3, 6.0)
	>>> circuit = builder.build()
	>>> circuit_path_voltage(circuit, [1, 2, 3, 4])
	6.0
	>>> circuit_path_voltage(circuit, [4, 3, 2, 1])
	-6.0

	'''
	acc = 0.0
	for s,t in vpath.edges(path):
		acc += circuit_edge_sign(circuit,s,t) * circuit.edge[s][t][EATTR_VOLTAGE]
	return acc

def circuit_path_resistance(circuit, path):
	'''
	Get the total resistance along a path.

	>>> import networkx as nx
	>>> builder = CircuitBuilder(nx.cycle_graph(5))
	>>> builder.make_resistor(2, 3, 5.0)
	>>> circuit = builder.build()
	>>> circuit_path_resistance(circuit, [1, 2, 3, 4])
	5.0
	'''
	return sum(circuit.edge[s][t][EATTR_RESISTANCE] for (s,t) in vpath.edges(path))

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

class cached_property:
	'''
	A member function decorator which provides a clean way to defer computation.

	A property-like object which is automatically backed by a member.
	A value may be assigned to it via ``=`` (like a property) or by explicitly calling
	 ``put(value)``.  To obtain its value, one must explicitly call ``get()``; if no
	 value has been stored, it will call the decorated function to compute a new value,
	 storing and returning it.

	Example:

	>>> class Foo:
	...   @cached_property
	...   def x(self):
	...     print('computing!!!')
	...     return 10  # return value gets stored
	...
	>>> foo = Foo()
	>>> foo.x.get()
	computing!!!
	10
	>>> foo.x.get()
	10
	>>> foo.x.put(3)
	>>> foo.x = 3    # sugared form of put()
	>>> foo.x.get()
	3
	>>> foo.x.invalidate()  # force recomputation
	>>> foo.x.get()
	computing!!!
	10
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
		'''
		Obtain the cached value (or compute a new one).

		If a value was previously computed with ``get()`` (or stored via
		``put()``), it will be returned.  Otherwise, the decorated member
		function will be invoked to compute a new value.
		'''
		if getattr(obj, self.member, self.sentinel) is self.sentinel:
			setattr(obj, self.member, self.func(obj))
		assert getattr(obj, self.member) is not self.sentinel
		return getattr(obj, self.member)

	def put(self, obj, value):
		'''
		Store a value directly to the cache.

		This is useful for providing optimized routines which can update a member
		without having to recompute it from scratch.  Rather than calling ``invalidate()``,
		``get()`` the value, perform the necessary transformations, and place it back
		with ``put()``.
		'''
		setattr(obj, self.member, value)

	def invalidate(self, obj):
		'''
		Delete the cached value, causing it to be recomputed on the next ``get()``.
		'''
		setattr(obj, self.member, self.sentinel)

# a cached_property bound to an instance
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
	def __init__(self, circuit, cyclebasis, cbupdater=None):
		if cbupdater is None:
			cbupdater = cyclebasis_provider.dummy_cbupdater()

		validate_circuit(circuit)

		self.__g = circuit.copy()
		self.__cbupdater = cbupdater

		self.__cbupdater.init(cyclebasis)

		# Invalidate everything
		self.__cyclebasis.invalidate()
		self.__cycles_from_edge.invalidate()
		self.__voltage_vector.invalidate()
		self.__resistance_matrix.invalidate()
		self.__cycle_currents.invalidate()

	def delete_node(self, v):
		'''
		Removes a vertex and all associated edges from the circuit.
		'''
		if not self.__g.has_node(v):
			raise KeyError('No such node: {}'.format(repr(v)))

		# update in-place
		self.__g.remove_node(v)
		self.__cbupdater.remove_vertex(self.__g, v)

		self.__cyclebasis.invalidate()
		self.__cycles_from_edge.invalidate()
		self.__voltage_vector.invalidate()
		self.__resistance_matrix.invalidate()
		self.__cycle_currents.invalidate()

	def multiply_edge_resistance(self, s, t, factor):
		'''
		Multiplies the resistance of an edge by a scalar factor.
		'''
		self.__g.edge[s][t][EATTR_RESISTANCE] *= factor

		self.__cyclebasis       # still valid!
		self.__cycles_from_edge # still valid!
		self.__voltage_vector   # still valid!
		self.__resistance_matrix.invalidate()
		self.__cycle_currents.invalidate()

	def assign_edge_resistance(self, s, t, value):
		'''
		Assigns a value to the resistance of an edge.
		'''
		self.__g.edge[s][t][EATTR_RESISTANCE] = value

		self.__cyclebasis       # still valid!
		self.__cycles_from_edge # still valid!
		self.__voltage_vector   # still valid!
		self.__resistance_matrix.invalidate()
		self.__cycle_currents.invalidate()

	# FIXME: Ick. This is here so that the node_deletion module can do what it needs.
	#             I don't want to expose the graph directly, nor do I want to make
	#             frequent copies due to its size... yet at the same time, a method
	#             like this really feels out of place!
	def node_neighbors(self, v):
		'''
		Get the immediate neighbors of a node.
		'''
		return self.__g.neighbors(v)

	def node_exists(self, v):
		'''
		Determine if a node exists in the circuit. (Boolean)
		'''
		return self.__g.has_node(v)

	def circuit(self):
		'''
		Get a copy of the current state of the circuit.
		'''
		return self.__g.copy()

	@cached_property
	def __cyclebasis(self):
		cb = self.__cbupdater.get_cyclebasis()

		# NOTE: this test is here because it is one of the only few paths that code
		#  reliably passes through where the current state of the modified cyclebasis
		#  and graph are both available.

		# FIXME: whatever happened to validate_cyclebasis?
		if len(cb) != len(nx.cycle_basis(self.__g)):

			# FIXME: This is an error (rather than assertion) due to an unresolved issue
			#  with the builder updater algorithm;  I CANNOT say with confidence that
			#  this will not occur. -_-
			raise RuntimeError('Cyclebasis has incorrect rank ({}, need {}).'.format(
				len(cb),len(nx.cycle_basis(self.__g))))

		return cb

	@cached_property
	def __cycles_from_edge(self):
		return compute_cycles_from_edge(self.__g, self.__cyclebasis.get())

	@cached_property
	def __voltage_vector(self):
		return compute_voltage_vector(self.__g, self.__cyclebasis.get())

	@cached_property
	def __resistance_matrix(self):
		return compute_resistance_matrix(self.__g, self.__cyclebasis.get(), self.__cycles_from_edge.get())

	@cached_property
	def __cycle_currents(self):
		return compute_cycle_currents(self.__resistance_matrix.get(), self.__voltage_vector.get(), self.__cyclebasis.get())

	def get_all_currents(self):
		'''
		Compute all currents in the circuit.

		The return value is a dict ``d`` such that ``d[s,t]`` (for an existing edge ``(s,t)``)
		is the (signed) current that flows from ``s`` to ``t``. It is guaranteed that
		``d[s,t] == -(d[t,s])``.
		'''
		# NOTE: of course, the guarantee that d[s,t] == -d[t,s] is written under the assumption
		#       that d[s,t] is not NaN
		d = compute_all_edge_currents(self.__g, self.__cycle_currents.get(), self.__cycles_from_edge.get())
		assert all(d[t,s] == -d[s,t] for s,t in d)
		return d

	def get_current(self, s, t):
		'''
		Compute signed current current flowing from ``s`` to ``t``.

		It is a ``KeyError`` if no such edge exists in the graph.
		'''
		if not self.__g.has_edge(s,t):
			raise KeyError('no such edge: {}'.format(repr((s,t))))
		return compute_single_edge_current(self.__g, self.__cycle_currents.get(), self.__cycles_from_edge.get(), s, t)

#------------------------------------------------------------

# Functions which actually compute stuff for MeshCurrentSolver.

# These are laid out as free functions so that all data dependencies are clearly
#  spelled out in the arguments.

def compute_cycles_from_edge(g, cyclebasis):
	cycles_from_edge = {e:[] for e in g.edges()}

	for pathI, path in enumerate(cyclebasis):
		for e in vpath.edges(path):
			sign = circuit_edge_sign(g, *e)

			ecycles = edictget(cycles_from_edge, e)
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

		ecycles = edictget(cycles_from_edge, e)

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
	ecycles = edictget(cycles_from_edge, (s,t))
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

