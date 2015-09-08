
from defect.circuit import CircuitBuilder
from defect.components.node_deletion import *
import unittest

from defect.circuit import MeshCurrentSolver
from defect.components.cyclebasis_provider import builder_cbupdater
import numpy as np

from defect import util

def edge_resistance(circuit, v1, v2):
	from defect.circuit import circuit_path_resistance
	return circuit_path_resistance(circuit, [v1, v2])

class FromInfoTests(unittest.TestCase):
	def dotest(self, obj):
		''' Test that an object can be recovered from ``from_info`` '''
		info = obj.info()
		obj2 = from_info(info)
		self.assertEqual(type(obj), type(obj2))
		self.assertEqual(obj, obj2)

	def test_all(self):
		# Hmmm. Feels as though I am liable to forget to add things to this list...
		self.dotest(annihilation())
		self.dotest(multiply_resistance(100., False))
		self.dotest(multiply_resistance(100., True))

class StandardTests(unittest.TestCase):

	def setUp(self):
		import networkx as nx
		n = 8
		g = nx.cycle_graph(n)
		builder = CircuitBuilder(g)

		cycle = list(range(n)) + [0]
		for s,t in util.window2(cycle):
			builder.make_resistor(s, t, 2.0)
		circuit = builder.build()

		self.solver = MeshCurrentSolver(circuit, [cycle], builder_cbupdater())
		self.edges = list(util.window2(cycle)) # all original edges in order

	def test_multiply(self):
		solver = self.solver
		deleter = multiply_resistance(4., False)
		# pick two nodes that share an edge to test non-idempotence
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assert np.allclose(resistances, [2., 2., 8., 32., 8., 2., 2., 2.])

	def test_assign(self):
		solver = self.solver
		deleter = multiply_resistance(4., True)
		# pick two nodes that share an edge to test idempotence
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assert np.allclose(resistances, [2., 2., 4., 4., 4., 2., 2., 2.])

	def test_remove(self):
		solver = self.solver
		deleter = annihilation()
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified node list
		mod_circuit = solver.circuit()
		self.assertListEqual(sorted(mod_circuit), [0,1,2,5,6,7])

