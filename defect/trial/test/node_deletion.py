
from defect.circuit import CircuitBuilder
from defect.trial.node_deletion import *
import unittest

from defect.circuit import MeshCurrentSolver
from defect.trial.cyclebasis_provider import builder_cbupdater
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
		self.dotest(annihilation(radius=3, single_defect=False))
		self.dotest(multiply_resistance(100., False, radius=2))
		self.dotest(multiply_resistance(100., True,  radius=4))

# Try to minimize coupling of the individual tests with DeletionMode's (fairly unstable)
#  API by just having this function extract out the one most important thing (a callback
#  for introducing a defect at a vertex)
def get_delete_cb(solver, deletionmode):
	deleter = deletionmode.deleter(solver.circuit())
	def f(v):
		return deleter.delete_one(solver, v, cannot_touch=[])
	return f

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
		delete = get_delete_cb(solver, multiply_resistance(4., False, radius=1))
		# pick two nodes that share an edge to test non-idempotence
		delete(3)
		delete(4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assertAllClose(resistances, [2., 2., 8., 32., 8., 2., 2., 2.])

	def test_assign(self):
		solver = self.solver
		delete = get_delete_cb(solver, multiply_resistance(4., True, radius=1))
		# pick two nodes that share an edge to test idempotence
		delete(3)
		delete(4)

		# inspect modified circuit resistances
		mod_circuit = solver.circuit()
		resistances = np.array([edge_resistance(mod_circuit, s, t) for (s,t) in self.edges])
		assertAllClose(resistances, [2., 2., 4., 4., 4., 2., 2., 2.])

	def test_remove(self):
		solver = self.solver
		delete = get_delete_cb(solver, annihilation(radius=1, single_defect=False))
		delete(3)
		delete(4)

		# inspect modified node list
		mod_circuit = solver.circuit()
		self.assertListEqual(sorted(mod_circuit), [0,1,2,5,6,7])

	# This test can be disabled if the behavior is no longer found desirable.
	if True:

		# tests that deleting a node with radius>1 still affects its neighborhood (acc.
		#  to the initial graph state) even if the targeted node no longer exists.
		def test_ghost_deletion(self):
			def remaining_nodes():
				return set(self.solver.circuit().nodes())

			delete = get_delete_cb(self.solver, annihilation(radius=3, single_defect=False))
			delete(1) # deletes nodes 7 0 1 2 3
			self.assertSetEqual(remaining_nodes(), set([4, 5, 6]))

			delete(2) # a node that no longer exists, which has a neighbor that still does
			self.assertSetEqual(remaining_nodes(), set([5, 6]))


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
		delete = get_delete_cb(self.solver, multiply_resistance(4., False, radius=3))

		self.check_all_resistances(2.)
		delete('a') # an inner vertex
		self.check_all_resistances(8.)
		delete('A') # an outer vertex
		self.check_all_resistances(32.)

	def test_assign_large(self):
		# radius large enough to affect all edges
		delete = get_delete_cb(self.solver, multiply_resistance(4., True, radius=3))

		self.check_all_resistances(2.)
		delete('a') # an inner vertex
		self.check_all_resistances(4.)
		delete('A') # an outer vertex
		self.check_all_resistances(4.)

	def test_multiply_small(self):
		# smaller radius so edges are left behind
		delete = get_delete_cb(self.solver, multiply_resistance(4., False, radius=2))

		delete('A') # outer vertex
		self.check_resistances(['Aa','ab','ac'], 8.) # affected edges
		self.check_resistances(['bB','cC','bc'], 2.) # unaffected edges

		delete('B') # the other outer vertices
		delete('C')
		self.check_resistances(['ab','bc','ca'], 32.) # affected twice
		self.check_resistances(['aA','bB','cC'], 8.)  # affected once

	def test_assign_small(self):
		# smaller radius so edges are left behind
		delete = get_delete_cb(self.solver, multiply_resistance(4., True, radius=2))

		delete('A') # outer vertex
		self.check_resistances(['Aa','ab','ac'], 4.) # affected edges
		self.check_resistances(['bB','cC','bc'], 2.) # unaffected edges

		delete('B') # the other outer vertices
		delete('C')
		self.check_resistances(['ab','bc','ca'], 4.) # affected twice
		self.check_resistances(['aA','bB','cC'], 4.) # affected once

	def test_remove_large(self):
		# radius large enough to affect all vertices
		delete = get_delete_cb(self.solver, annihilation(radius=4, single_defect=False))
		delete('A') # an outer vertex

		# there should be no nodes left!
		self.assertSetEqual(set(self.solver.circuit()), set())

	def test_remove_small(self):
		# smaller radius so that some vertices are left behind
		delete = get_delete_cb(self.solver, annihilation(radius=3, single_defect=False))

		def remaining_nodes():
			return set(self.solver.circuit().nodes())

		# pick "overlapping" spots, mostly to make sure we don't get an
		#  unexpected exception for some reason (I can think of mistakes
		#  that might cause one if I ever were to implement a stateful
		#  part to deletion methods like I have for selection)
		delete('A') # an outer vertex
		self.assertSetEqual(remaining_nodes(), {'B', 'C'})

		delete('B')
		self.assertSetEqual(remaining_nodes(), {'C'})

		delete('C')
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

