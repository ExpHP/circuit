# Methods for introducing "defects" into the circuit.
# These objects provide a callback which modifies the provided graph to represent
#  the introduction of a defect at a given node.

__all__ = [
	'annihilation',
	'multiply_resistance',
]

class annihilation:
	''' Represents a defect by removing the vertex from the underlying graph structure. '''
	def deletion_func(self, solver, v):
		solver.delete_node(v)

	def info(self):
		return {'mode': 'direct removal'}

class multiply_resistance:
	''' Represents a defect by modifying the resistances of edges connected to the vertex. '''
	def __init__(self, factor, idempotent):
		assert isinstance(factor, float)
		assert isinstance(idempotent, bool)
		self.factor = factor
		self.idempotent = idempotent

	def deletion_func(self, solver, v):
		if self.idempotent:
			solver.assign_nearby_resistances(v, self.factor)
		else:
			solver.multiply_nearby_resistances(v, self.factor)

	def info(self):
		return {
			'mode': 'multiply resistance',
			'factor': self.factor,
			'idempotent': self.idempotent,
		}

