
import sys
import math
import random
import time
try:
	import cProfile as profile
except ImportError:
	import profile

import networkx as nx
import numpy as np

from multiprocessing import Pool
import multiprocessing_dill

from circuit import Circuit
from graphs.planar_cycle_basis import minimal_cycle_basis as planar_cycle_basis
from resistances_common import *

import pick


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input',   type=str, help='.gml or .gml.gz file')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--jobs', '-j', type=int, default=1, help='number of trials to run in parallel')
	parser.add_argument('--num-trials', '-n', type=int, default=10, help='number of trials to do total')
	args = parser.parse_args(sys.argv[1:])

	f = lambda : run_trial_fpath(20, args.input, verbose=args.verbose)
	print(run_parallel(f, threads=args.jobs, times=args.num_trials))

# Runs a nullary function the specified number of times and collects the results into an array.
# Each process will be given a unique random seed
def run_parallel(f,*,threads,times):

	baseseed = time.time()
	seeds = [baseseed + i for i in range(times)]

	def run_with_seed(seed):
		random.seed(seed)
		return f()

	p = Pool(threads)
	return multiprocessing_dill.map(p, run_with_seed, seeds)

def run_trial_fpath(steps, path, *, verbose=False):
	return run_trial_nx(steps, read_graph(path), verbose=verbose)

def run_trial_nx(steps, g, *, verbose=False):
	if verbose:
		print('Starting')

	initial_g = g.copy()

	step_current = []
	for step in range(steps):
		t = time.time()

		# the big heavy calculation!
		step_current.append(compute_current_planar_nx(g))

		# delete some nodes
		deleted = get_nodes_to_delete(g, initial_g)
		remove_nodes_from(g, deleted)

		if verbose:
			print('step: ', step, 'time: ', time.time() - t)

	return step_current

def get_nodes_to_delete(g, initial_g):
	return [pick.uniform(get_deletable_nodes(g))]

def compute_current_planar_nx(g):
	circuit = circuit_from_nx(g)

	xs = {v:g.node[v]['x'] for v in g}
	ys = {v:g.node[v]['y'] for v in g}

	measure_s = g.graph[GATTR_MEASURE_SOURCE]
	measure_t = g.graph[GATTR_MEASURE_TARGET]

	measure_e = circuit.arbitrary_edge(measure_s, measure_t)

	return compute_current_planar(circuit, measure_e, xs, ys)

def compute_current_planar(circuit, measured_edge, xs, ys):
	es = {e: circuit.edge_endpoints(e) for e in circuit.edges()}
	vertex_cyclebasis = planar_cycle_basis(circuit.vertices(), es, xs, ys)
	edge_cyclebasis = [to_edge_path(circuit, cycle) for cycle in vertex_cyclebasis]

	return circuit.compute_currents(edge_cyclebasis)[measured_edge]

# note: visualizing nx graphs
#nx.draw_networkx(g, pos={v:(xs[v],ys[v]) for v in g})
#plt.show()

def read_graph(path):
	g = nx.read_gpickle(path)
	return g

def get_deletable_nodes(g):
	return set(v for v in g if g.node[v][VATTR_REMOVABLE])

# Prefer g.remove_node over g.remove_nodes_from.
# (the latter silently ignores invalid nodes, which is a problem
#  on the off-chance that it is accidentally supplied with a
#  node of an iterable type (i.e. string) instead of a list)
def remove_nodes_from(g, nodes):
	for v in nodes:
		g.remove_node(v)

def circuit_from_nx(g):
	circuit = Circuit()
	circuit.add_vertices(iter(g))
	for (v1,v2) in g.edges():
			eattr = g.edge[v1][v2]
			s = eattr[EATTR_SOURCE]
			t = other_endpoint_nx((v1,v2), s)

			circuit.add_component(s, t,
				resistance = eattr[EATTR_RESISTANCE],
				voltage = eattr[EATTR_VOLTAGE],
			)
	return circuit

def other_endpoint_nx(e, s):
	(v1,v2) = e
	if s == v1: return v2
	elif s == v2: return v1
	else:
		raise ValueError('{} is not an endpoint of the edge ({},{})'.format(s,v1,v2))

def to_edge_path(circuit, vertex_path):
	# HACK should be SingleGraph.get_edge instead of BaseGraph.get_arbitrary_edge
	#     but I haven't implemented the former class yet
	# or better yet just get rid of my silly pointless incomplete graph library >_>
	return [circuit.arbitrary_edge(u,v) for (u,v) in window2(vertex_path)]

# A scrolling 2-element window on an iterator
def window2(it):
	it = iter(it)
	prev = next(it)
	for x in it:
		yield (prev,x)
		prev = x

if __name__ == '__main__':
	main()
