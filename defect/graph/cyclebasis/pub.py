
# cyclebasis/pub.py -- public functions to be exported from defect.graph.cyclebasis

# (nothing interesting here; just delegates to implementations located elsewhere).

from . import _planar
from defect.graph.cyclebasis.builder import CycleBasisBuilder
import defect.filetypes.internal as fileio

import networkx as nx
from defect.util import unzip_dict

class planar:
	'''
	Construct cyclebasis from planar embedding.

	May be invoked in the following ways:
	 * ``planar(g, pos)``, where ``pos`` is a dict of ``{node: (x,y)}``
	    describing a planar embedding of the nodes in ``g``.
	 * ``planar.from_gpos(g, path)``, where ``path`` is a filepath to
	    a ``.planar.gpos`` file that provides an embedding for ``g``.
	'''
	def __init__(self, g, pos):
		xs,ys = unzip_dict(pos)
		return _planar.planar_cycle_basis_nx(g, xs, ys)

	@classmethod
	def from_gpos(cls, g, path):
		pos = fileio.gpos.read_gpos(path)
		return cls(g, pos)

def from_file(path):
	'''
	Read a cyclebasis from a .cycles file.

	This always puts the cycles into the format expected by the
	defect trial (in contrast to ``defect.filetypes.internal.cycles.read_cycles``
	which has extraneous options)
	'''
	# TODO we should validate the cyclebasis against g here
	# (I thought I had a method which did this, but can't find it...)
	return fileio.cycles.read_cycles(path, repeatfirst=True)

def last_resort(g):
	'''
	Produce a (low-quality) cyclebasis using a fast and reliable method.

	This should be avoided for graphs of any decent size, as it tends
	to produce cyclebases which have a lot of overlap, leading to a
	dense resistance matrix.  (if you have a dense resistance matrix,
	you're gonna have a bad time)
	'''
	cycles = list(nx.cycle_basis(g))
	for c in cycles:
		c.append(c[0]) # make loop
	return cycles

#-----------------------------------------------------------

# cbupdaters, which are provided to CurrentMeshSolver so it can... update the cbs.

class planar_cbupdater:
	'''
	Allows one to update a cyclebasis in response to changes in the graph.

	Do not use this one.
	'''
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		self.cycles = _planar.without_vertex(self.cycles, v)
	def get_cyclebasis(self):
		return self.cycles

class builder_cbupdater:
	'''
	Allows one to update a cyclebasis in response to changes in the graph.

	This one keeps track of basis elements and linear dependencies by
	building and maintaining a bit matrix in REF form.
	'''
	def init(self, cycles):
		self.builder = CycleBasisBuilder.from_basis_cycles(cycles)
	def remove_vertex(self, g, v):
		self.builder.remove_vertex(v)
	def get_cyclebasis(self):
		return self.builder.cycles

class dummy_cbupdater:
	'''
	Does NOT allow one to update a cyclebasis in response to changes in the graph,
	and instead throws exceptions if you try to do anything remotely change-y.

	This exists because ``CurrentMeshSolver`` always expects a ``cbupdater`` even
	if it does not modify the graph.  With a ``dummy_cbupdater``, it is possible
	to use ``CurrentMeshSolver`` to compute currents only for the initial state.
	'''
	def init(self, cycles):
		self.cycles = cycles
	def remove_vertex(self, g, v):
		raise NotImplementedError("dummy_cbupdater")
	def get_cyclebasis(self):
		return self.cycles

