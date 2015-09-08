
from abc import ABCMeta, abstractmethod

__all__ = [
	'DeletionMode',
	'annihilation',
	'multiply_resistance',
	'from_info',
]

class DeletionMode(metaclass=ABCMeta):
	'''
	A component of the defect trial runner.  Controls how defects are represented.

	``DeletionMode``s are stateless and safe to reuse between trials.
	'''
	# FIXME: cannot_touch is a temporary HACK until I can think of something better.
	#  It basically says: "do not modify these nodes or any edge connected to them
	#   under any circumstances... um, unless it's an edge and you're deleting the
	#   other node the edge is connected to, in which case it's totally fine that
	#   the edge will be deleted as well. Yeppers."
	# >_>
	# This had to be added to protect the battery, because with the radius option,
	#  defects are now capable of affecting nodes outside the set of `choices`!
	@abstractmethod
	def deletion_func(self, solver, v, cannot_touch):
		''' Add a defect to a ``MeshCurrentSolver``. '''
		pass

	@abstractmethod
	def info(self):
		''' Get representation for ``results.json``. '''
		pass

	@classmethod
	@abstractmethod
	def from_info(cls, info):
		''' Recreate from info(). '''
		pass

	def __eq__(self, other):
		# (assumes info() provides all of the meaningful content (and only that content))
		return type(self) is type(other) and self.info() == other.info()

# It is a deliberate design choice that all deletion modes with a "radius" argument
#  define it in such a way that this is the smallest possible value.
# This is to avoid making it needlessly difficult to switch between methods.
MIN_VALID_RADIUS = 1

#--------------------------------------------------------

class annihilation(DeletionMode):
	'''
	Removes the vertex from the underlying graph structure.

	``radius=1`` deletes a single vertex, ``radius=2`` deletes a vertex
	and its neighbors, etc.
	'''
	def __init__(self, radius):
		assert isinstance(radius, int)
		assert radius >= MIN_VALID_RADIUS
		self.radius = radius

	def deletion_func(self, solver, v, cannot_touch):
		cannot_touch = set(cannot_touch)

		if not solver.node_exists(v):
			return

		vs = _neighborhood(solver, v, maxdist=self.radius-1)
		for node in vs:
			if node in cannot_touch:
				continue
			if solver.node_exists(node):
				solver.delete_node(node)

	def info(self):
		return {
			'mode': 'direct removal',
			'radius': self.radius,
		}

	@classmethod
	def from_info(cls, info):
		return cls(radius=info['radius'])

#--------------------------------------------------------

class multiply_resistance(DeletionMode):
	'''
	Modifies the resistances of edges connected to the vertex.

	``idempotent`` toggles between multiplication (False) and assignment (True).

	``radius=1`` affects the edges around a single vertex. ``radius=2`` affects up
	to 2 edges away, and so on.
	'''
	def __init__(self, factor, idempotent, radius):
		assert isinstance(factor, float)
		assert isinstance(idempotent, bool)
		assert isinstance(radius, int)
		assert radius >= MIN_VALID_RADIUS
		self.factor = factor
		self.idempotent = idempotent
		self.radius = radius

	def deletion_func(self, solver, v, cannot_touch):
		cannot_touch = set(cannot_touch)

		vs = _neighborhood(solver, v, maxdist=self.radius-1)
		es = _collect_edges(solver, vs)
		for s,t in es:
			if s in cannot_touch or t in cannot_touch:
				continue
			if self.idempotent:
				solver.assign_edge_resistance(s, t, self.factor)
			else:
				solver.multiply_edge_resistance(s, t, self.factor)

	def info(self):
		return {
			'mode': 'multiply resistance',
			'factor': self.factor,
			'idempotent': self.idempotent,
			'radius': self.radius,
		}

	@classmethod
	def from_info(cls, info):
		return cls(
			factor = info['factor'],
			idempotent = info['idempotent'],
			radius = info['radius']
		)

#--------------------------------------------------------

# Even if modestrings are changed, the old names should remain here
#  in addition to the new names, for backwards compatibility.
CLASSES_FROM_MODESTRS = {
	'direct removal': annihilation,
	'multiply resistance': multiply_resistance,
}

def from_info(info):
	''' Recreate a DeletionMode from its info. '''
	modestr = info['mode']
	cls = CLASSES_FROM_MODESTRS[modestr]
	return cls.from_info(info)

#--------------------------------------------------------

def _neighborhood(solver, v, maxdist):
	''' Get all vertices up to ``maxdist`` edges from ``v``. '''
	if maxdist < 0:
		return set()

	out = set([v])
	if maxdist > 0:
		for nbr in solver.node_neighbors(v):
			# The following optimization is NOT valid when we are traversing in DFS order.
			# If such an optimization REALLY is needed, rewrite this algo as a BFS instead!
			#if nbr in out: continue  # <--- NOTE: DO NOT DO THIS!!!!!!!

			out.update(_neighborhood(solver, nbr, maxdist-1))

	return out

def _collect_edges(solver, vs):
	'''
	Get all of the edges involving any of the vs.

	Returns an iterable that contains each such (undirected) edge exactly once.
	The type of the vertices must form a total order.
	'''
	es = []
	for s in vs:
		es.extend((s,t) for t in solver.node_neighbors(s))

	# canonicalize edges so that s < t, to make duplicates obvious
	es = [(s,t) if s<t else (t,s) for (s,t) in es]

	# remove duplicates
	es = set(es)
	return es

