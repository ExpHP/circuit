
# Two versions of XorBasis, c one should be faster
try:
	import graph.cyclebasis.cXorBasis as xorbasis
except ImportError:
	import graph.cyclebasis.pyXorBasis as xorbasis

import graph.path as vpath

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

# XXX REPLACE
#		assert len(list(self.basis.get_linearly_dependent_ids())) == 0 # class invariant

		return success

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

