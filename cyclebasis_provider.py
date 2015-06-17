
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

	def info(self): return {} # XXX

# Loaded from separate file
class from_file:
	def __init__(self, path): pass # XXX
	def is_planar(self): return False
	def new_cyclebasis(self, g): return None # XXX
	def info(self): return {} # XXX

# Networkx built-in cycle_basis method
class last_resort:
	def is_planar(self): return False
	def new_cyclebasis(self, g): return None # XXX
	def info(self): return {} # XXX

