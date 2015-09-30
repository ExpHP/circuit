
from . import _planar
from defect.graph.cyclebasis.builder import CycleBasisBuilder
import defect.filetypes.internal as fileio

import networkx as nx
from defect.util import unzip_dict

# TODO kill the ``new_cyclebasis`` methods and make these just functions

# Uses planar embedding info from graph
class planar:
	def __init__(self, pos):
		''' pos: dict of v: (x,y) '''
		self.__pos = pos

	@classmethod
	def from_gpos(cls, path):
		pos = fileio.gpos.read_gpos(path)
		return cls(pos)

	def new_cyclebasis(self, g):
		xs,ys = unzip_dict(self.__pos)
		return _planar.planar_cycle_basis_nx(g, xs, ys)

# Loaded from separate file
class from_file:
	def __init__(self, path):
		self.path = path

	def new_cyclebasis(self, g):
		# TODO we should validate the cyclebasis against g here
		# (I thought I had a method which did this, but can't find it...)
		return fileio.cycles.read_cycles(self.path, repeatfirst=True)

# Networkx built-in cycle_basis method
class last_resort:
	def cbupdater(self):
		return builder_cbupdater()

	def new_cyclebasis(self, g):
		cycles = list(nx.cycle_basis(g))
		for c in cycles:
			c.append(c[0]) # make loop
		return cycles

#-----------------------------------------------------------

# cbupdaters, which are provided to CurrentMeshSolver so it can... update the cbs.

# for planar graphs only
class planar_cbupdater:
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		self.cycles = _planar.without_vertex(self.cycles, v)
	def get_cyclebasis(self):
		return self.cycles

# via builder.CycleBasisBuilder
class builder_cbupdater:
	def init(self, cycles):
		self.builder = CycleBasisBuilder.from_basis_cycles(cycles)
	def remove_vertex(self, g, v):
		self.builder.remove_vertex(v)
	def get_cyclebasis(self):
		return self.builder.cycles

# For things that don't need to update the graph, such as... pretty any client of
#  CurrentMeshSolver other than the defect introduction algo.
class dummy_cbupdater:
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		raise NotImplementedError("dummy_cbupdater")
	def get_cyclebasis(self):
		return self.cycles
