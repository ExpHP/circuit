
import json
import networkx as nx
import graph.cyclebasis.planar
from graph.cyclebasis.builder import CycleBasisBuilder
from util import unzip_dict
import filetypes.internal as fileio

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
	def __init__(self, pos):
		''' pos: dict of v: (x,y) '''
		self.__pos = pos

	@classmethod
	def from_gpos(cls, path):
		pos = fileio.gpos.read_gpos(path)
		return cls(pos)

	def cbupdater(self):
		return builder_cbupdater()

	def new_cyclebasis(self, g):
		xs,ys = unzip_dict(self.__pos)
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
		# TODO we should validate the cyclebasis against g here
		# (I thought I had a method which did this, but can't find it...)
		return fileio.cycles.read_cycles(self.path, repeatfirst=True)

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

# Wraps a previously-computed cyclebasis in a cyclebasis_provider.
# (FIXME: this is here as a quick hack so that other scripts (e.g. tests) can invoke run_trial_nx
#  with their own cyclebasis; it would be nicer if running a trial didn't require a cbprovider)
class from_existing:
	def __init__(self, cyclebasis):
		self.cycles = cyclebasis

	def cbupdater(self):
		return builder_cbupdater()

	def new_cyclebasis(self, g):
		return self.cycles

	def info(self):
		return {'mode': 'already computed'}


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
		self.builder.remove_vertex(v)
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
