
import networkx as nx
import unittest
import random

from defect.circuit import *
from defect.components import cyclebasis_provider

class MiscCurrentTests(unittest.TestCase):
	# A circuit with multiple connected components that each contain cycles;
	#  this is something that can arise naturally when deleting nodes from
	#  a larger graph.
	def test_two_separate_loops(self):
		g = nx.Graph()
		g.add_path('abca')
		g.add_path('xyzx')
		builder = CircuitBuilder(g)
		builder.make_battery('a', 'b', 5.0)
		builder.make_resistor('b', 'c', 1.0)
		builder.make_resistor('c', 'a', 1.0)
		builder.make_battery('x', 'y', 5.0)
		builder.make_resistor('y', 'z', 1.0)
		builder.make_resistor('z', 'x', 1.0)

		circuit = builder.build()
		currents = compute_circuit_currents(circuit)

		for s,t in ('ab', 'bc', 'ca', 'xy', 'yz', 'zx'):
			assertNear(currents[s,t], +2.5)
			assertNear(currents[t,s], -2.5)

	# A circuit with loops that share an edge.
	# This is an extremely basic test for how the resistance matrix is generated;
	#  even a simple circuit such as this has cycles which are "linearly dependent"
	#  (the two loops, and their "sum")
	def test_two_loop_circuit(self):
		# {0: 5.0, 1: -0.99999999999999956, 2: 4.0, 3: -5.0, 4: 0.99999999999999956}
		g = nx.Graph()
		g.add_path(['Up','Rt','Dn','Lt','Up'])
		g.add_edge('Dn','Up')
		builder = CircuitBuilder(g)
		builder.make_battery('Dn', 'Lt', 28.0)
		builder.make_battery('Dn', 'Rt',  7.0)
		builder.make_resistor('Up', 'Dn', 2.0)
		builder.make_resistor('Up', 'Lt', 4.0)
		builder.make_resistor('Up', 'Rt', 1.0)

		circuit = builder.build()
		currents = compute_circuit_currents(circuit)

		assertNear(currents['Dn','Lt'], +5.0)
		assertNear(currents['Dn','Rt'], -1.0)
		assertNear(currents['Up','Dn'], +4.0)
		assertNear(currents['Up','Lt'], -5.0)
		assertNear(currents['Up','Rt'], +1.0)

	# `MeshCurrentSolver.get_current` and `compute_circuit_currents` should provide
	#  consistent results
	def test_get_current_consistency(self):
		g = nx.gnm_random_graph(5,16)
		builder = CircuitBuilder(g)
		for s,t in g.edges():
			builder.make_component(s, t, resistance=random.random(), voltage=random.random())
		circuit = builder.build()

		# freestanding function results
		currents = compute_circuit_currents(circuit)

		# class results
		cbprovider = cyclebasis_provider.last_resort()
		solver = MeshCurrentSolver(circuit, cbprovider.new_cyclebasis(circuit), cbprovider.cbupdater())

		# consistent?
		for s,t in g.edges():
			assertNear(solver.get_current(s,t), currents[s,t])
			assertNear(solver.get_current(t,s), currents[t,s])

def assertNear(a,b,eps=1e-7):
	assert abs(a-b) < eps

