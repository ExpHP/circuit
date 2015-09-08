
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
		self.dotest(annihilation(radius=3))
		self.dotest(multiply_resistance(100., False, radius=2))
		self.dotest(multiply_resistance(100., True,  radius=4))

# Tests operating on an 8-node cycle graph.
# Useful for very basic tests of functionality.
class CycleGraphTests(unittest.TestCase):

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
		deleter = multiply_resistance(4., False, radius=1)
		# pick two nodes that share an edge to test non-idempotence
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assertAllClose(resistances, [2., 2., 8., 32., 8., 2., 2., 2.])

	def test_assign(self):
		solver = self.solver
		deleter = multiply_resistance(4., True, radius=1)
		# pick two nodes that share an edge to test idempotence
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assertAllClose(resistances, [2., 2., 4., 4., 4., 2., 2., 2.])

	def test_remove(self):
		solver = self.solver
		deleter = annihilation(radius=1)
		deleter.deletion_func(solver, 3)
		deleter.deletion_func(solver, 4)

		# inspect modified node list
		mod_circuit = solver.circuit()
		self.assertListEqual(sorted(mod_circuit), [0,1,2,5,6,7])

# Tests operating on a 6-node graph that looks like the Eiffel Tower:
#                 (A)
#                  |
#                 (a)
#                 / \
#               (b)-(c)
#               /     \
#             (B)     (C)
# Designed to test a couple of aspects of deletion modes that have an
#  adjustable radius.  Some features exploted in the tests:
#  * A 3-cycle, used to make sure that nodes encountered through two
#     different paths aren't inadvertently affected twice.
#  * Edges stemming off the 3-cycle, which might be missed by some
#     poorly-designed traversal methods. (...such as my initial
#     implementation X_X)
class EiffelTowerTests(unittest.TestCase):

	def setUp(self):
		import networkx as nx
		g = nx.Graph()
		g.add_cycle('abc')               # triangle
		g.add_edges_from(['aA','bB','cC']) # antennae
		builder = CircuitBuilder(g)

		for s,t in g.edges():
			builder.make_resistor(s, t, 2.0)
		circuit = builder.build()

		cycles = [['a','b','c','a']]
		self.solver = MeshCurrentSolver(circuit, cycles, builder_cbupdater())
		self.edges = g.edges()

	def check_resistances(self, edges, expected):
		# check that all given edges have the specified resistance
		mod_circuit = self.solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in edges])
		assertAllClose(resistances, [expected])

	def check_all_resistances(self, expected):
		self.check_resistances(self.edges, expected)

	def test_multiply_large(self):
		# radius large enough to affect all edges
		deleter = multiply_resistance(4., False, radius=3)

		self.check_all_resistances(2.)
		deleter.deletion_func(self.solver, 'a') # an inner vertex
		self.check_all_resistances(8.)
		deleter.deletion_func(self.solver, 'A') # an outer vertex
		self.check_all_resistances(32.)

	def test_assign_large(self):
		# radius large enough to affect all edges
		deleter = multiply_resistance(4., True, radius=3)

		self.check_all_resistances(2.)
		deleter.deletion_func(self.solver, 'a') # an inner vertex
		self.check_all_resistances(4.)
		deleter.deletion_func(self.solver, 'A') # an outer vertex
		self.check_all_resistances(4.)

	def test_multiply_small(self):
		# smaller radius so edges are left behind
		deleter = multiply_resistance(4., False, radius=2)

		deleter.deletion_func(self.solver, 'A') # outer vertex
		self.check_resistances(['Aa','ab','ac'], 8.) # affected edges
		self.check_resistances(['bB','cC','bc'], 2.) # unaffected edges

		deleter.deletion_func(self.solver, 'B') # the other outer vertices
		deleter.deletion_func(self.solver, 'C')
		self.check_resistances(['ab','bc','ca'], 32.) # affected twice
		self.check_resistances(['aA','bB','cC'], 8.)  # affected once

	def test_assign_small(self):
		# smaller radius so edges are left behind
		deleter = multiply_resistance(4., True, radius=2)

		deleter.deletion_func(self.solver, 'A') # outer vertex
		self.check_resistances(['Aa','ab','ac'], 4.) # affected edges
		self.check_resistances(['bB','cC','bc'], 2.) # unaffected edges

		deleter.deletion_func(self.solver, 'B') # the other outer vertices
		deleter.deletion_func(self.solver, 'C')
		self.check_resistances(['ab','bc','ca'], 4.) # affected twice
		self.check_resistances(['aA','bB','cC'], 4.) # affected once

	def test_remove_large(self):
		# radius large enough to affect all vertices
		deleter = annihilation(radius=4)
		deleter.deletion_func(self.solver, 'A') # an outer vertex

		# there should be no nodes left!
		self.assertSetEqual(set(self.solver.circuit()), set())

	def test_remove_small(self):
		# smaller radius so that some vertices are left behind
		deleter = annihilation(radius=3)

		def remaining_nodes():
			return set(self.solver.circuit().nodes())

		# pick "overlapping" spots, mostly to make sure we don't get an
		#  unexpected exception for some reason (I can think of mistakes
		#  that might cause one if I ever were to implement a stateful
		#  part to deletion methods like I have for selection)
		deleter.deletion_func(self.solver, 'A') # an outer vertex
		self.assertSetEqual(remaining_nodes(), {'B', 'C'})

		deleter.deletion_func(self.solver, 'B')
		self.assertSetEqual(remaining_nodes(), {'C'})

		deleter.deletion_func(self.solver, 'C')
		self.assertSetEqual(remaining_nodes(), set())

def assertAllClose(array1, array2):
	array1 = np.array(array1)
	array2 = np.array(array2)
	if not np.allclose(array1, array2):
		# try to at least provide SOME info before failing.
		# hopefully the mismatch doesn't get ellided out.
		print('First array:')
		print(array1)
		print()
		print('Second array:')
		print(array2)
		assert False, 'Arrays not close!'

