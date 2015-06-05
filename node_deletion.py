# Methods for introducing "defects" into the circuit.
# These objects provide a callback which modifies the provided graph to represent
#  the introduction of a defect at a given node.

__all__ = [
	'annihilation',
]

class annihilation:
	def deletion_func(self, solver, v):
		solver.delete_node(v)

	def info(self):
		return {'mode': 'direct removal'}

