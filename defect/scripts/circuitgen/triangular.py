#!/usr/bin/env python3

import os
import sys
import math
import argparse

import numpy as np
import networkx as nx

import defect.trial
from defect.circuit import save_circuit, CircuitBuilder
from defect.util import zip_dict, zip_matching_length, window2
import defect.filetypes.internal as fileio

def main(prog, argv):
	parser = argparse.ArgumentParser(prog=prog)
	parser.add_argument('rows', metavar='LENGTH', type=int)
	parser.add_argument('cols', metavar='WIDTH', type=int)
	parser.add_argument('--output', '-o', type=str, required=True, help='.circuit output file')
	parser.add_argument('--cb', action='store_true', help='generate .cycles')

	args = parser.parse_args(argv)

	values = make_circuit(args.rows, args.cols)
	save_output(args.output, args.cb, *values)


def make_circuit(nrows, ncols):
	# Nodes
	gridvs = [['g@{},{}'.format(row,col) for col in range(ncols)] for row in range(nrows)]

	colxs = np.arange(1.*ncols)
	rowys = np.arange(1.*nrows) * (3**0.5)/2

	gridxs = np.outer(np.ones(colxs.shape), colxs)
	gridys = np.outer(rowys, np.ones(rowys.shape))

	# zigzag
	for i in range(1, nrows, 2):
		gridxs[i,:] += 0.5

	xs = {v: x for v,x in zip(flat_iter(gridvs), gridxs.flat)}
	ys = {v: y for v,y in zip(flat_iter(gridvs), gridys.flat)}

	# Connector nodes
	xdim = max(xs.values()) - min(xs.values())
	ydim = max(ys.values()) - min(ys.values())

	topv = 'top'
	xs[topv] = -0.1 * xdim
	ys[topv] = +1.1 * ydim

	botv = 'bot'
	xs[botv] = -0.1 * xdim
	ys[botv] = -0.1 * ydim

	#----------------
	# Circuit object
	g = nx.Graph()

	resistors = []

	# These edges:  -
	for row in gridvs:
		resistors.extend(window2(row))

	# Edges between nodes in same grid column
	# (these are / edges or \ edges depending on the row,
	#  because every other row is shifted to the right)
	for col in transpose_iter(gridvs):
		resistors.extend(window2(col))

	# \ edges between different columns (an odd row above an even row)
	for i in range(1, len(gridvs), 2):
		resistors.extend(zip(gridvs[i][:-1], gridvs[i-1][1:]))

	# / edges between different columns (an even row above an odd row)
	for i in range(2, len(gridvs), 2):
		resistors.extend(zip(gridvs[i][1:], gridvs[i-1][:-1]))

	g.add_edges_from(resistors)
	g.add_edges_from((topv, v) for v in gridvs[-1])
	g.add_edges_from((botv, v) for v in gridvs[0])
	g.add_edge(botv, topv)

	# add circuit properties
	builder = CircuitBuilder(g)
	for s,t in resistors:
		builder.make_resistor(s, t, 1.0)

	builder.make_battery(botv, topv, 1.0)

	circuit = builder.build();

	# finish
	no_defect = []
	measure_edge = (botv, topv)
	return circuit, xs, ys, measure_edge, no_defect

def save_output(path, do_cb, g, xs, ys, measure_edge, no_defect):
	save_circuit(g, path)

	basename = os.path.splitext(path)[0]

	gpos_path = basename + '.planar.gpos'
	pos = zip_dict(xs, ys)
	fileio.gpos.write_gpos(pos, gpos_path)

	config = defect.trial.Config()
	config.set_measured_edge(*measure_edge)
	config.set_no_defect(no_defect)
	config.save(basename + '.defect.toml')

	if do_cb:
		from defect.graph.cyclebasis.planar import planar_cycle_basis_nx
		cycles = planar_cycle_basis_nx(g, xs, ys)
		fileio.cycles.write_cycles(cycles, basename + '.cycles')

# a flattened iterator over a (singly-)nested iterable.
def flat_iter(lst):
	for item in lst:
		yield from item

def transpose_iter(lst):
	return zip_matching_length(*lst)

if __name__ == '__main__':
	prog, *argv = sys.argv
	main(prog, argv)

