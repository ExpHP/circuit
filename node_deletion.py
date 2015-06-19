# Methods for introducing "defects" into the circuit.
# These objects provide a callback which modifies the provided graph to represent
#  the introduction of a defect at a given node.

__all__ = [
	'annihilation',
	'scalar_multiply',
]

class annihilation:
	''' Represents a defect by removing the vertex from the underlying graph structure. '''
	def deletion_func(self, solver, v):
		solver.delete_node(v)

	def info(self):
		return {'mode': 'direct removal'}

class scalar_multiply:
	''' Represents a defect by multiplying the resistance of each edge connected to
	    the vertex by a constant factor. '''
	def __init__(self, factor):
		self.factor = float(factor)

	def deletion_func(self, solver, v):
		solver.multiply_nearby_resistances(v, self.factor)

	def info(self):
		return {
			'mode': 'multiply resistance',
			'factor': self.factor,
		}
