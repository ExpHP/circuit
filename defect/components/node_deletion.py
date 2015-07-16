
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
	@abstractmethod
	def deletion_func(self, solver, v):
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

#--------------------------------------------------------

class annihilation(DeletionMode):
	''' Removes the vertex from the underlying graph structure. '''
	def deletion_func(self, solver, v):
		solver.delete_node(v)

	def info(self):
		return {'mode': 'direct removal'}

	@classmethod
	def from_info(cls, info):
		return cls()

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

	@classmethod
	def from_info(cls, info):
		return cls(
			factor = info['factor'],
			idempotent = info['idempotent'],
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

def test_from_info(obj):
	''' Test that an object can be recovered from ``from_info`` '''
	info = obj.info()
	obj2 = from_info(info)
	assert type(obj) is type(obj2)
	assert obj == obj2

test_from_info(annihilation())
test_from_info(multiply_resistance(100., False))
test_from_info(multiply_resistance(100., True))

