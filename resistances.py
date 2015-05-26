
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
import json

from multiprocessing import Pool
import multiprocessing_dill

from circuit import Circuit
from graphs.planar_cycle_basis import planar_cycle_basis
from resistances_common import *

import node_selection

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='.pickle file of networkx graph')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--jobs', '-j', type=int, default=1, help='number of trials to run in parallel')
	parser.add_argument('--trials', '-t', type=int, default=10, help='number of trials to do total')
	parser.add_argument('--steps', '-s', type=int, default=100, help='number of steps per trial')
	parser.add_argument('--output-json', '-o', type=str, default=None, help='output file')

	args = parser.parse_args(sys.argv[1:])

	#selector = node_selection.uniform()
	selector = node_selection.by_deleted_neighbors([1,10**3,10**4,10**7])

	f = lambda : run_trial_fpath(
		steps = args.steps,
		path = args.input,
		selection_func = selector.selection_func,
		verbose = args.verbose
	)

	info = {}

	info['selection_mode'] = selector.info()

	info['process_count'] = args.jobs

	info['time_started'] = int(time.time())
	info['trials'] = run_parallel(f, threads=args.jobs, times=args.trials)
	info['time_finished'] = int(time.time())

	if args.output_json is not None:
		s = json.dumps(info)
		with open(args.output_json, 'w') as f:
			f.write(s)

#	visualize_selection_func(read_graph(args.input), selection_funcs.uniform, 500)
#	visualize_selection_func(read_graph(args.input), selection_funcs.near_deleted, 500)

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

def run_trial_fpath(steps, path, selection_func, *, verbose=False):
	trial_info = run_trial_nx(steps, read_graph(path), selection_func, verbose=verbose)
	trial_info['filepath'] = path
	return trial_info

#def run_trial_nx(steps, g, selection_func, *, verbose=False):
	#profile.runctx('run_trial_nx_(steps,g,selection_func,verbose=verbose)',locals(),globals(),sort='tot')

def run_trial_nx(steps, g, selection_func, *, verbose=False):
	if verbose:
		print('Starting')

	initial_g = g.copy()

	trial_info = {
		'graph': graph_info_nx(g),
		'steps': {'runtime':[], 'current':[], 'deleted':[]},
	}

	choices = get_deletable_nodes(g)

	step_info = trial_info['steps']
	for step in range(steps):
		t = time.time()

		# the big heavy calculation!
		current = compute_current_planar_nx(g)

		# delete a node
		deleted = selection_func(choices, g, initial_g)
		choices.remove(deleted)
		g.remove_node(deleted)

		runtime = time.time() - t

		step_info['runtime'].append(runtime)
		step_info['current'].append(current)
		step_info['deleted'].append([deleted]) # list in anticipation of multiple deletions

		if verbose:
			print('step:', step, 'time:', runtime, 'current:', current)

	return trial_info

def visualize_selection_func(g, selection_func, nsteps):
	initial_g = g.copy()
	for _ in range(nsteps):
		deleted = selection_func(g, initial_g)
		remove_nodes_from(g, deleted)

	import matplotlib.pyplot as plt
	from functools import partial

	fig, ax = plt.subplots()
	ax.set_aspect('equal')
	xs = nx.get_node_attributes(initial_g, 'x')
	ys = nx.get_node_attributes(initial_g, 'y')
	pos={v:(xs[v],ys[v]) for v in initial_g}

	draw = partial(nx.draw_networkx, g, pos, False, ax=ax, node_size=50)
	draw(edgelist=initial_g.edges(), nodelist=[])
	draw(edgelist=[], nodelist=set(initial_g)-set(g), node_color='r')
	draw(edgelist=[], nodelist=set(g), node_color='g')

	plt.show()

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

def read_graph(path):
	g = nx.read_gpickle(path)
	return g

def get_deletable_nodes(g):
	return set(v for v in g if g.node[v][VATTR_REMOVABLE])

def graph_info_circuit(circuit):
	return {
		'num_vertices': circuit.num_vertices(),
		'num_edges':    circuit.num_edges(),
	}

def graph_info_nx(g):
	return {
		'num_vertices': g.number_of_nodes(),
		'num_edges':    g.number_of_edges(),
	}

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
