
import math

import circuit

import numpy as np

# (these top two should be flipped x/y)
# Column numbers: (zigzag dimension)
#
#     0  0        0  0      cell row
#  1        1  1        1       0
#     2  2        2  2          1
#  3        3  3        3       2
#     4  4        4  4          3
# (5)       5  5       (5)
#
# Row numbers: (armchair direction)
#
#     0  1        2  3
#  0        1  2        3
#     0  1        2  3
#  0        1  2        3
#     0  1        2  3
#           1  2
#
# If you were to plot these indices and the bonds between, you'd get a brick-like layout:
#
#  *-*-*-*-*-*-*
#  |   |   |   |
#  *-*-*-*-*-*-*-*
#    |   |   |   |
#  *-*-*-*-*-*-*-*
#  |   |   |   |
#  *-*-*-*-*-*-*
#
class Vertex():
	def __init__(self, row, col):
		self._row = row
		self._col = col

	@property
	def row(self): return self._row
	@property
	def col(self): return self._col

	# geometric position of the point on the hexagonal grid that the vertex represents
	@property
	def x(self):
		return 0.5 * math.sqrt(3) * self.col
	@property
	def y(self):
		y = 1.5 * self.row # baseline height of row
		y += 0.5 * ((self.row + self.col + 1) % 2) # adjust every other point for zigzag
		return y

	def _value_tuple(self):
		return (self.row, self.col)
	def __eq__(self, other):
		return self._value_tuple() == other._value_tuple()
	def __hash__(self):
		return hash(self._value_tuple())
	def __repr__(self):
		return 'Vertex({},{})'.format(self.row, self.col)




def make_hex_bridge_circuit(cellrows, cellcols):
	assert cellrows > 0
	assert cellcols > 0
	
	g = circuit.Circuit()

	nrows = 2*(cellcols+1)
	ncols = cellrows+1

	g.add_vertices(Vertex(row,col) for row in range(nrows) for col in range(ncols))

	# let's lay some bricks!

	# horizontal edges
	for row in range(nrows):
		# all the way across
		for col in range(ncols-1):
			g.add_resistor(Vertex(row,col), Vertex(row,col+1), 1)

	# vertical edges
	for row in range(nrows-1):
		# take every other column (alternating between rows)
		for col in range(row % 2, ncols, 2):
			g.add_resistor(Vertex(row+1,col), Vertex(row, col), 1)

	return g

g = make_hex_bridge_circuit(10,6)
g.add_battery(Vertex(0,3), Vertex(13,3), 100.)
print(g.compute_currents())


def do_visualize():
	import matplotlib.pyplot as plt
	from matplotlib import collections as mc

	g = make_hex_bridge_circuit(10,6)
	xs = np.array([v.x for v in g.vertices()])
	ys = np.array([v.y for v in g.vertices()])
	g.add_resistor(Vertex(0,5), Vertex(13,5), 0.)

	lines = []
	for e in g.edges():
		u,v = g.edge_endpoints(e)
		lines.append(((u.x,u.y), (v.x,v.y)))
	lc = mc.LineCollection(lines, colors='k',linewidths=2)

	print(xs)

	fig,ax = plt.subplots()
	ax.set_aspect('equal')
	ax.scatter(xs,ys,s=25.)
	ax.add_collection(lc)
	plt.show()

