
import json
import networkx as nx
import graph.cyclebasis.planar
from graph.cyclebasis.builder import CycleBasisBuilder

__all__ = [
	'planar',
	'from_file',
	'last_resort',
	'planar_cbupdater',
	'builder_cbupdater',
	'dummy_cbupdater',
]

# Uses planar embedding info from graph
class planar:
	def cbupdater(self):
		return planar_cbupdater()

	def new_cyclebasis(self, g):
		xs = {v:g.node[v]['x'] for v in g}
		ys = {v:g.node[v]['y'] for v in g}

		return graph.cyclebasis.planar.planar_cycle_basis_nx(g, xs, ys)

	def info(self):
		return {'mode': 'from planar embedding'}

# Loaded from separate file
class from_file:
	def __init__(self, path):
		self.path = path

	def cbupdater(self):
		return builder_cbupdater()

	def new_cyclebasis(self, g):
		with open(self.path) as f:
			s = f.read()
		return json.loads(s)

	def info(self):
		return {
			'mode': 'from file',
			'path': self.path,
		}

# Networkx built-in cycle_basis method
class last_resort:
	def cbupdater(self):
		return builder_cbupdater()

	def new_cyclebasis(self, g):
		cycles = list(nx.cycle_basis(g))
		for c in cycles:
			c.append(c[0]) # make loop
		return cycles

	def info(self):
		return {'mode': 'networkx builtin'}


#-----------------------------------------------------------

# cbupdaters, which are provided to CurrentMeshSolver so it can... update the cbs.

# for planar graphs only
class planar_cbupdater:
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		self.cycles = graph.cyclebasis.planar.without_vertex(self.cycles, v)
	def get_cyclebasis(self):
		return self.cycles

# via builder.CycleBasisBuilder
class builder_cbupdater:
	def init(self, cycles):
		self.builder = CycleBasisBuilder.from_basis_cycles(cycles)
	def remove_vertex(self, g, v):
		self.builder.remove_vertex(g, v)
	def get_cyclebasis(self):
		return self.builder.cycles

# For things that don't need to update the graph, such as... pretty any client of
#  CurrentMeshSolver other than the defect introduction algo.
# (well, now it sounds kind of silly that CurrentMeshSolver requires one)
# (TODO: is this a sign of some greater issue?)
class dummy_cbupdater:
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		raise NotImplementedError("dummy_cbupdater")
	def get_cyclebasis(self):
		return self.cycles
