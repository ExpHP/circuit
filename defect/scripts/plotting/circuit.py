#!/usr/bin/env python3

# defect -- view-circuit.py
# Visualization tool for circuits that doesn't necessarily aim for
#  anything presentable, and is mostly for verifying the output
#  of a circuit-generating script.

import sys
import os
from argparse import ArgumentParser

import networkx as nx
import matplotlib.pyplot as plt

import defect.circuit as dc
import defect.filetypes.internal.gpos as gpos

from functools import partial

def main(prog=None, *argv):
	# (support no-arg invocation for setup.py)
	if prog is None:
		return main(*sys.argv)
	prog = os.path.split(prog)[1]

	parser = ArgumentParser(prog, description='Visualize a .circuit file')
	parser.add_argument('infiles', nargs='*', help='input files, among which must be a .circuit file and a .gpos file')

	args = parser.parse_args(argv)

	# parse list of infiles
	remaining = list(args.infiles)
	def accept_infile_with_extension(ext):
		ext = ext.lower()
		files = [x for x in remaining if x.lower().endswith(ext)]
		if len(files) > 1: die('too many %s files!', ext)
		if len(files) < 1: parser.error('need a {} file!'.format(ext))

		path, = files
		remaining.remove(path)
		return path

	circuitpath = accept_infile_with_extension('.circuit')
	gpospath    = accept_infile_with_extension('.gpos')

	for path in remaining:
		warn("Unused argument: '%s'", path)

	# load files
	def try_read(path, readfunc):
		try:
			result = readfunc(path)
		except (IOError, OSError) as e: die('Cannot read %s: %s', path, e)
		except e: die('Error parsing %s: %s', path, e)
		return result

	circuit = try_read(circuitpath, dc.load_circuit)
	pos     = try_read(gpospath, gpos.read_gpos)

	validate_pos(circuit, pos)

	# go go go gooooo
	draw_circuit(circuit, pos)

# check that the pos file is compatible with the circuit
def validate_pos(circuit, pos):

	# sets of nodes representing possible issues
	missingvs = set(circuit) - set(pos)
	extravs   = set(pos) - set(circuit)

	# collect error messages for each issue
	errors = []
	for (badvs, errmsg) in [
		(missingvs, ' * gpos file is missing vertices (%(cnt)s total): %(lst)s'),
		(extravs,   ' * gpos file has extra vertices (%(cnt)s total): %(lst)s'),
	]:
		if len(badvs) == 0:
			continue  # no issue

		# ERROR TIME!
		# name just a few of the bad vertices
		MAX_SHOWN = 3
		lst = list(badvs)[:MAX_SHOWN]
		lstmsg = ', '.join(repr(x) for x in lst)

		# elide the rest
		if len(badvs) > MAX_SHOWN:
			lstmsg += ', ...'
		errors.append(errmsg % {'cnt':len(badvs), 'lst':lstmsg})

	if errors:
		lines = ['Gpos does not match circuit!'] + errors
		die('\n'.join(lines))

def draw_circuit(g, pos):
	fig, ax = plt.subplots()
	ax.set_aspect('equal')

	resistors = set(e for e in g.edges() if dc.circuit_path_resistance(g, e) != 0.0)
	batteries = set(e for e in g.edges() if dc.circuit_path_voltage(g, e) != 0.0)

	# both a resistor AND an edge?
	# (such edges are probably a mistake, so make sure they're given a separate color)
	wildedges = resistors & batteries
	resistors -= wildedges
	batteries -= wildedges

	# undeniably plain edges
	wires = set(g.edges()) - resistors - batteries - wildedges

	# bind some common arguments
	draw_nodes = partial(nx.draw_networkx_nodes, g, ax=ax, pos=pos)
	draw_edges = partial(nx.draw_networkx_edges, g, ax=ax, pos=pos)

	draw_nodes(node_size=50.)
	draw_edges(edgelist=resistors, edge_color='b', width=2., label='resistor')
	draw_edges(edgelist=batteries, edge_color='r', width=2., label='battery')
	draw_edges(edgelist=wildedges, edge_color='m', width=2., label='unusual')
	draw_edges(edgelist=wires,     edge_color='k', width=1., label='plain wire')

	ax.legend(loc=1)
	plt.show()

def info(msg, *args):
	print(msg % args)

def warn(msg, *args):
	print('Warning: ' + (msg % args), file=sys.stderr)

def die(msg, *args, code=1):
	print('\nFatal: ' + (msg % args), file=sys.stderr)
	sys.exit(code)

if __name__ == '__main__':
	main()

