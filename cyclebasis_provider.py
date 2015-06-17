
import networkx as nx
import graph.cyclebasis.planar

__all__ = [
	'planar',
	'from_file',
	'last_resort',
]

# Uses planar embedding info from graph
class planar:
	def is_planar(self): return True

	def new_cyclebasis(self, g):
		xs = {v:g.node[v]['x'] for v in g}
		ys = {v:g.node[v]['y'] for v in g}

		return graph.cyclebasis.planar.planar_cycle_basis_nx(g, xs, ys)

	def info(self):
		return {'mode': 'from planar embedding'}

# Loaded from separate file
class from_file:
	def is_planar(self): return False

	def __init__(self, path):
		self.path = path

	def new_cyclebasis(self, g):
		raise RuntimeError('not implemented!') # XXX

	def info(self):
		return {
			'mode': 'from file',
			'path': self.path,
		}

# Networkx built-in cycle_basis method
class last_resort:
	def is_planar(self): return False

	def new_cyclebasis(self, g):
		cycles = list(nx.cycle_basis(g))
		for c in cycles:
			c.append(c[0]) # make loop
		return cycles

	def info(self):
		return {'mode': 'networkx builtin'}

