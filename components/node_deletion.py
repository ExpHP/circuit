
from abc import ABCMeta, abstractmethod

__all__ = [
	'DeletionMode',
	'annihilation',
	'multiply_resistance',
]

class DeletionMode(metaclass=ABCMeta):
	'''
	A component of the defect trial runner.  Controls how defects are represented.

	``DeletionMode``s are stateless and safe to reuse between trials.
	'''
	@abstractmethod
	def deletion_func(self, solver, v):
		''' Add a defect to a ``MeshCurrentSolver``. '''
		pass
	@abstractmethod
	def info(self):
		''' Get representation for ``results.json``.

		Return value is a ``dict`` with at minimum a ``'mode'`` value (a string). '''
		pass

#--------------------------------------------------------

class annihilation(DeletionMode):
	''' Removes the vertex from the underlying graph structure. '''
	def deletion_func(self, solver, v):
		solver.delete_node(v)

	def info(self):
		return {'mode': 'direct removal'}

#--------------------------------------------------------

class multiply_resistance(DeletionMode):
	''' Modifies the resistances of edges connected to the vertex. '''
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

