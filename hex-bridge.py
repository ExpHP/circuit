
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

class Vertex:
	def __init__(self, row, col):
		self._row = row
		self._col = col

		# object used for hashing and equality testing.
		# stored as a member in response to profiling tests
		self._value_tuple = (row, col)

	@property
	def row(self): return self._row
	@property
	def col(self): return self._col

	# geometric position of the point on the hexagonal grid that the vertex represents
	@property
	def x(self):
		if self == self.TOP_CONNECTOR: return -4.0
		elif self == self.BOT_CONNECTOR: return -1.0
		else:
			return 0.5 * math.sqrt(3) * self.col
	@property
	def y(self):
		if self in (self.TOP_CONNECTOR, self.BOT_CONNECTOR):
			return -3.0
		else:
			y = 1.5 * self.row # baseline height of row
			y += 0.5 * ((self.row + self.col + 1) % 2) # adjust every other point for zigzag
			return y

	def __eq__(self, other):
		return self._value_tuple == other._value_tuple
	def __hash__(self):
		return hash(self._value_tuple)
	def __repr__(self):
		return 'Vertex({},{})'.format(self.row, self.col)

# special constants
Vertex.TOP_CONNECTOR = Vertex(-1,0)
Vertex.BOT_CONNECTOR = Vertex(-1,1)



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

	# HACK
	g.add_vertices([Vertex.TOP_CONNECTOR, Vertex.BOT_CONNECTOR])
	g.add_battery(Vertex.TOP_CONNECTOR, Vertex.BOT_CONNECTOR, 100.0)
	for col in range(1,ncols,2):
		g.add_resistor(Vertex(0,col), Vertex.BOT_CONNECTOR, 1.0)
	for col in range(nrows%2+1, ncols, 2):
		g.add_resistor(Vertex(nrows-1,col), Vertex.TOP_CONNECTOR, 1.0)

	return g

g = make_hex_bridge_circuit(10,6)
#print(g.compute_currents())


def run_trial(nrows, ncols, steps):
	g = make_hex_bridge_circuit(nrows,ncols)

	battery = g.arbitrary_edge(Vertex.TOP_CONNECTOR, Vertex.BOT_CONNECTOR)

	step_currents = []
	for stepnum in range(steps):
		step_currents.append(g.compute_currents()[battery])

		r = g.random_vertex()
		while r in (Vertex.TOP_CONNECTOR, Vertex.BOT_CONNECTOR):
			r = g.random_vertex()

		g.delete_vertices([r])
	return step_currents

import matplotlib.pyplot as plt
from matplotlib import collections as mc

def show_trial(nrows, ncols, steps, fitupto):
	x = range(steps)

	fig,ax = plt.subplots()
	ys = [run_trial(nrows, ncols, steps) for _ in range(10)]
	for y in ys:
		ax.plot(x,y)
	plt.show()

	avg = np.sum(ys,axis=0)/len(ys)
	fig,ax = plt.subplots()
	ax.set_aspect('equal')
	ax.plot(x,avg)

	p = np.polyfit(x[:fitupto],avg[:fitupto],1)
	ax.plot(x,np.polyval(p,x), ':g')

	plt.show()

def bench_trial(nrows, ncols, steps):
	ys = [run_trial(nrows, ncols, steps) for _ in range(10)]

def do_visualize(g):

	xs = np.array([v.x for v in g.vertices()])
	ys = np.array([v.y for v in g.vertices()])

	lines = []
	for e in g.edges():
		u,v = g.edge_endpoints(e)
		lines.append(((u.x,u.y), (v.x,v.y)))
	lc = mc.LineCollection(lines, colors='k',linewidths=2)

	fig,ax = plt.subplots()
	ax.set_aspect('equal')
	ax.scatter(xs,ys,s=25.)
	ax.add_collection(lc)
	plt.show()

import cProfile
#show_trial(20,6,steps=100,fitupto=20)
cProfile.run('bench_trial(10,6,steps=20)',sort='tottime')
#do_visualize(g)
