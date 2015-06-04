# Methods for introducing "defects" into the circuit.
# These objects provide a callback which modifies the provided graph to represent
#  the introduction of a defect at a given node.

import graphs.planar_cycle_basis as planar_cycle_basis

__all__ = [
	'annihilation',
]

class annihilation:
	def deletion_func(self, g, cyclebasis, v):
		g = graph_copy_without_vertex_nx(g, v)
		cyclebasis = planar_cycle_basis.without_vertex(cyclebasis, v)
		return g, cyclebasis

	def info(self):
		return {'mode': 'direct removal'}

def graph_copy_without_vertex_nx(g, v):
	vs = set(g)
	vs.remove(v)
	return g.subgraph(vs)

