
# Two versions of XorBasis, c one should be faster

# TODO: Python implementation is unavaible until its API is brought back up to speed
#try:
#	import graph.cyclebasis.cXorBasis as xorbasis
#except ImportError:
#	import graph.cyclebasis.pyXorBasis as xorbasis
import defect.ext.cXorBasis as xorbasis

import graph.path as vpath

from util import assertRaises

__all__ = [
	'CycleBasisBuilder',
]

class CycleBasisBuilder:
	def __init__(self):
		self.edge_mapper = EdgeIndexMapper()
		self.cycles_by_id = {}
		self.basis = xorbasis.XorBasisBuilder()

	@property
	def cycles(self):
		return self.cycles_by_id.values()

	# Constructs a CycleBasisBuilder from a set of cycles known to be linearly independent.
	# This computes the underlying matrix through a more efficient method than repeated calls
	#  to add_if_linearly_independent will provide.
	@classmethod
	def from_basis_cycles(cls, vcycles):

		self = cls()

		vcycles = list(vcycles)
		if any(c[0] != c[-1] for c in vcycles):
			raise RuntimeError('Expected cycles with repeated first vertex!')
		ecycles = list(map(self.edge_mapper.map_path, vcycles))
		identities = self.basis.add_many(ecycles)

		if len(self.basis.get_zero_sums()) > 0:
			raise RuntimeError("from_basis_cycles() was provided linearly dependent cycles!")

		for identity,cycle in zip(identities, vcycles):
			self.cycles_by_id[identity] = cycle

		assert len(vcycles) == len(self.cycles_by_id)
		return self

	# If the provided cycle is linearly independent from the cycles already in
	#  the cyclebasis, adds the cycle and returns True.
	# Otherwise, returns False.
	def add_if_independent(self, cycle):
		if not vpath.is_cycle(cycle):
			raise ValueError('CycleBasisBuilder was provided a non-cycle')
		cycle = list(cycle)
		edgeids = self.edge_mapper.map_path(cycle)

		success, identity = self.basis.add_if_linearly_independent(edgeids)

		if success:
			assert identity not in self.cycles_by_id
			self.cycles_by_id[identity] = cycle

		return success

	# Updates the cycle basis to account for the removal of a vertex from the graph.
	def remove_vertex(self, v):
		import networkx as nx

		badpaths = self.__pop_all_with_vertex(v)

		if len(badpaths) == 0: # degenerate case
			return

		g = nx.Graph()
		for path in badpaths:
			g.add_path(path)

		g.remove_node(v)

		rebuilt = nx.cycle_basis(g)
		for cycle in rebuilt:
			cycle.append(cycle[0])
			self.add_if_independent(cycle)

	# Removes all cycles with vertex v and returns them
	def __pop_all_with_vertex(self, v):
		invalidated = [(i,path) for (i,path) in self.cycles_by_id.items() if v in path]

		if len(invalidated) == 0: # degenerate case for zip
			return []

		ids, paths = zip(*invalidated)

		for i in ids:
			del self.cycles_by_id[i]

		# remove the corresponding rows from the rref bit matrix
		self.basis.remove_ids(ids)

		return paths

#----------------------

class EdgeIndexMapper:
	def __init__(self):
		self.edges = []
		self.edge_indices = {}

	def map_path(self, path):
		result = []
		for s,t in vpath.edges(path):
			if (s,t) not in self.edge_indices:
				self.edge_indices[s,t] = len(self.edges)
				self.edge_indices[t,s] = len(self.edges)
				self.edges.append((s,t))
			result.append(self.edge_indices[s,t])
		return result

_cbb1 = CycleBasisBuilder()
assert _cbb1.add_if_independent('abca')
assert _cbb1.add_if_independent('dcbd')
assert not _cbb1.add_if_independent('acdba')
assert not _cbb1.add_if_independent('abdca')

# CycleBasisBuilder with degenerate cycle basis
assertRaises(RuntimeError, CycleBasisBuilder.from_basis_cycles, ['1231', '4234', '42134'])
