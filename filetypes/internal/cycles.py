
'''
A JSON-based format for a list of cycles.

This is intended to supplement an existing graph file by providing
cyclebasis cycles.  Cycles are stored as an array of arrays, where
each inner array is a list of unique nodes ``[v1, v2, ..., vN]``
(where each node is an integer or string) describing a cycle with
N edges (note v1 != vN), listed in order of traversal (such that
the edges (v1,v2), (v2,v3), ... (vN,v1) all exist).

The internal representation is a list of lists of nodes. One can
optionally specify for the first vertex to be repeated at the end,
(i.e. ``[v1, v2, ..., vN, v1]``) which is extremely useful for
iteration over edges.

Note that while this format is sufficient for describing cycles
in both directed and undirected graphs, it does not provide enough
information to pinpoint edges in a multigraph.

The cycles need not actually form a cyclebasis or be unique.  In
fact, because this file is saved independently of the graph, the
methods in this module are *incapable* of verifying the accuracy or
completeness of a cyclebasis.  You should validate cyclebases
yourself in code that has access to both the cycles and the graph.
'''

import json

HIGHEST_VERSION = 1

# TODO tests

def write_cycles(cycles, path):
	'''
	Save a cycles file.

	In the input cycles, either all must repeat the first vertex at
	the end, or none of them should.
	'''
	cycles = list(map(list, cycles))
	if len(cycles) > 0:
		pred = _repeats_first_vertex
		repeatfirst = pred(cycles[0])
		if any(pred(x) != repeatfirst for x in cycles):
			raise ValueError('Cycle storage scheme is inconsistent '
				"(some repeat the first vertex, others don't)")

		# Canonicalize by not repeating
		if repeatfirst:
			cycles = [x[:-1] for x in cycles]

	if any(len(set(x)) != len(x) for x in cycles):
		raise ValueError('A cycle has a repeated vertex.')

	d = {
		'formatver':  HIGHEST_VERSION,
		'cycles':     cycles,
	}
	with open(path, 'w') as f:
		json.dump(d, f)


def read_cycles(path, repeatfirst=False):
	'''
	Load a cycles file.

	If ``repeatfirst`` is ``True``, each cycle returned will have the first vertex
	repeated at the end, to assist in iteration over edges.
	'''
	with open(path) as f:
		d = json.load(f)

	if d['formatver'] > HIGHEST_VERSION:
		raise RuntimeError('Unsupported file format version {}'.format(d['formatver']))

	cycles = d['cycles']
	if any(len(set(x)) != len(x) for x in cycles):
		raise RuntimeError('A cycle has a repeated vertex.')

	if repeatfirst:
		for cycle in cycles:
			cycle.append(cycle[0])
	return cycles


def _repeats_first_vertex(cycle):
	# Second part of the condition accounts for self-loops
	#   ( we want False for [x],  True for [x, x] )
	return (cycle[0] == cycle[-1]) and len(cycle) > 1

