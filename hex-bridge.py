
import sys
import math
import argparse

import numpy as np
import networkx as nx

from resistances_common import *

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

parser = argparse.ArgumentParser()
parser.add_argument('rows', metavar='LENGTH', type=int)
parser.add_argument('cols', metavar='WIDTH', type=int)
parser.add_argument('--output', '-o', type=str, required=True, help='.gml or .gml.gz output file')

args = parser.parse_args(sys.argv[1:])

# Total number of rows/cols of vertices
def hex_grid_dims(cellrows, cellcols):
	return (
		2*cellrows + 1,
		cellcols + 1,
	)

def hex_grid_xy_arrays(cellrows, cellcols):
	nrows,ncols = hex_grid_dims(cellrows, cellcols)

	# 2d arrays containing row or column of each node
	rows = np.outer(range(nrows), [1]*ncols)
	cols = np.outer([1]*nrows, range(ncols))

	xs = 0.5 * math.sqrt(3) * cols

	ys = 1.5 * rows # baseline height
	ys += 0.5 * ((rows + cols + 1) % 2)  # zigzag across row

	return xs, ys

def grid_label(row,col):
	return 'grid@{},{}'.format(row,col)

# a flattened iterator over a (singly-)nested iterable.
def flat_iter(lst):
	for item in lst:
		yield from item

# a flattened iterator over the results of calling a function f
#  (which produces an iterable) on each element of an iterable.
def flat_map(f, lst):
	for item in lst:
		yield from f(item)

def hex_bridge_grid_circuit(gridvs):
	g = nx.Graph()

	g.add_nodes_from(flat_iter(gridvs))

	nrows,ncols = np.shape(gridvs)

	# horizontal edges
	for row in range(nrows):
		# all the way across
		for col in range(ncols-1):
			add_resistor(g, gridvs[row][col], gridvs[row][col+1], 1.0)

	# vertical edges
	for row in range(nrows-1):
		# take every other column (alternating between rows)
		for col in range(row % 2, ncols, 2):
			add_resistor(g, gridvs[row+1][col], gridvs[row][col], 1.0)

	return g

def add_wire(g, s, t):
	g.add_edge(s, t)
	g.edge[s][t][EATTR_RESISTANCE] = 0.0
	g.edge[s][t][EATTR_VOLTAGE]    = 0.0
	g.edge[s][t][EATTR_SOURCE]     = s

def add_resistor(g, s, t, resistance):
	add_wire(g,s,t)
	g.edge[s][t][EATTR_RESISTANCE] = 1.0

def add_battery(g, s, t, voltage):
	add_wire(g,s,t)
	g.edge[s][t][EATTR_VOLTAGE] = voltage

def validate_graph_attributes(g):
	if len(g) == 0: return
	if g.number_of_edges() == 0: return

	vattr_iter = iter(g.node.values())
	eattr_iter = flat_map(lambda d: d.values(), g.edge.values())

	expected_vattr = next(vattr_iter)
	expected_eattr = next(eattr_iter)

	for vattr in vattr_iter:
		diff = set(vattr) ^ set(expected_vattr)
		if len(diff) != 0:
			raise RuntimeError('node attributes {} are set on some nodes but not others'.format(diff))

	for eattr in eattr_iter:
		diff = set(eattr) ^ set(expected_eattr)
		if len(diff) != 0:
			raise RuntimeError('edge attributes {} are set on some edges but not others'.format(diff))

cellrows = args.rows
cellcols = args.cols

nrows,ncols = hex_grid_dims(cellrows,cellcols)

# Grid nodes
gridvs = [[grid_label(row,col) for col in range(ncols)] for row in range(nrows)]

gridxs, gridys = hex_grid_xy_arrays(cellrows,cellcols)
xs = {v: x for v,x in zip(flat_iter(gridvs), gridxs.flat)}
ys = {v: y for v,y in zip(flat_iter(gridvs), gridys.flat)}

# Connector nodes
topv = 'top'
xs[topv] = -2.0
ys[topv] = max(gridys.flat) + 1.0

botv = 'bot'
xs[botv] = -2.0
ys[botv] = -1.0

# Circuit object
g = hex_bridge_grid_circuit(gridvs)

g.add_nodes_from([topv, botv])

# Link top/bot with battery
add_battery(g, botv, topv, 1.0)

#----------------
# connect top/bot to nodes in graph
# due to zigzag pattern this is not entirely straightforward

# first column for "true" top/bottom rows
botstart = 1
topstart = (nrows+1) % 2

# doublecheck
assert gridys[0][botstart]  < gridys[0][1-botstart]
assert gridys[-1][topstart] > gridys[-1][1-topstart]

for v in gridvs[0][botstart::2]:
	add_wire(g,v,botv)

for v in gridvs[-1][topstart::2]:
	add_wire(g,v,topv)

deletable = {v: True for v in g}
deletable[topv] = False
deletable[botv] = False

# remove numpy type information from floats
xs = {v:float(x) for v,x in xs.items()}
ys = {v:float(x) for v,x in ys.items()}

nx.set_node_attributes(g, VATTR_X, xs)
nx.set_node_attributes(g, VATTR_Y, ys)
nx.set_node_attributes(g, VATTR_REMOVABLE, deletable)

# set edge that will have its current measured
g.graph[GATTR_MEASURE_SOURCE] = botv
g.graph[GATTR_MEASURE_TARGET] = topv

validate_graph_attributes(g)

nx.write_gpickle(g, args.output)

