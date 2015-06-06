
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

from circuit import MeshCurrentSolver, CircuitBuilder
import graphs.planar_cycle_basis as planar_cycle_basis
from resistances_common import *

import node_selection
import node_deletion

import graphs.vertexpath as vpath

import util

# TODO: maybe implement subparsers for these, and put in the node_selection/deletion
#   modules since these need to be updated for each new mode
SELECTION_MODES = {
	'uniform':  node_selection.uniform(),
	'bigholes': node_selection.by_deleted_neighbors([1,10**3,10**4,10**7]),
}
DELETION_MODES = {
	'remove':   node_deletion.annihilation(),
}

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='.gpickle file of networkx graph')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--jobs', '-j', type=int, default=1, help='number of trials to run in parallel')
	parser.add_argument('--trials', '-t', type=int, default=10, help='number of trials to do total')
	parser.add_argument('--steps', '-s', type=int, default=100, help='number of steps per trial')
	parser.add_argument('--output-json', '-o', type=str, default=None, help='output file')
	parser.add_argument('--output-pstats', '-P', type=str, default=None, help='Record profiling info (implies --jobs 1)')
	parser.add_argument('--selection-mode', '-S', type=str, required=True, choices=SELECTION_MODES, help='TODO')
	parser.add_argument('--deletion-mode', '-D', type=str, required=True, choices=DELETION_MODES, help='TODO')

	args = parser.parse_args(sys.argv[1:])

	if (args.output_pstats is not None and args.jobs != 1):
		print('error: --output-pstats/-P is limited to --jobs 1',file=sys.stderr)
		print('In other words: No multiprocess profiling!',file=sys.stderr)
		sys.exit(1)

	# save the user some grief; fail early if output paths are not writable
	for path in (args.output_json, args.output_pstats):
		if path is not None:
			error_if_not_writable(path)

	selector = SELECTION_MODES[args.selection_mode]
	deletor = DELETION_MODES[args.deletion_mode]

	cmd_once = lambda : run_trial_fpath(
		steps = args.steps,
		graphpath = args.input,
		selection_func = selector.selection_func,
		deletion_func  = deletor.deletion_func,
		verbose = args.verbose,
	)

	if args.jobs == 1:
		cmd_all = lambda: run_sequential(cmd_once, times=args.trials)
	else:
		cmd_all = lambda: run_parallel(cmd_once, threads=args.jobs, times=args.trials)

	if args.output_pstats is not None:
		assert args.jobs == 1
		cmd_all = wrap_with_profiling(args.output_pstats, cmd_all)

	info = {}

	info['selection_mode'] = selector.info()
	info['defect_mode'] = deletor.info()

	info['process_count'] = args.jobs
	info['profiling_enabled'] = (args.output_pstats is not None)

	info['time_started'] = int(time.time())
	info['trials'] = cmd_all() # do eeeet
	info['time_finished'] = int(time.time())

	assert isinstance(info['trials'], list)

	if args.output_json is not None:
		s = json.dumps(info)
		with open(args.output_json, 'w') as f:
			f.write(s)

# NOTE unintentional side-effect: creates an empty file if nothing exists
def error_if_not_writable(path):
	try:
		with open(path, 'a') as f:
			pass
	except IOError as e:
		print("Error: Could not verify '{}' as writable:".format(path), file=sys.stderr)
		print(str(e), file=sys.stderr)
		sys.exit(1)

#	visualize_selection_func(read_graph(args.input), selection_funcs.uniform, 500)
#	visualize_selection_func(read_graph(args.input), selection_funcs.near_deleted, 500)

def run_sequential(f,*,times):
	return [f() for _ in range(times)]

def run_parallel(f,*,threads,times):

	# Give each trial a unique seed
	baseseed = time.time()
	seeds = [baseseed + i for i in range(times)]

	def run_with_seed(seed):
		random.seed(seed)
		return f()

	p = Pool(threads)
	return multiprocessing_dill.map(p, run_with_seed, seeds)

def run_trial_fpath(steps, graphpath, selection_func, deletion_func, *, verbose=False):
	return run_trial_nx(steps, read_graph(graphpath), selection_func, deletion_func, verbose=verbose)

def wrap_with_profiling(pstatsfile, f):
	def wrapped(*args, **kwargs):
		p = profile.Profile()
		p.enable()
		result = f(*args, **kwargs)
		p.disable()

		try:
			p.dump_stats(pstatsfile)
		except IOError as e: # not worth losing our return value over
			print('Warning: could not write pstats. ({})'.format(str(e)), file=sys.stderr)

		return result
	return wrapped

def run_trial_nx(steps, g, selection_func, deletion_func, *, verbose=False):
	if verbose:
		print('Starting')

	initial_g = g.copy()

	trial_info = {
		'graph': graph_info_nx(g),
		'steps': {'runtime':[], 'current':[], 'deleted':[]},
	}

	measured_edge = get_measured_edge(g)
	choices = get_deletable_nodes(g)
	solver  = MeshCurrentSolver(g, cyclebasis_planar_nx(g), is_planar=True)

	past_selections = []

	step_info = trial_info['steps']
	for step in range(steps):
		t = time.time()

		# the big heavy calculation!
		current = solver.get_current(*measured_edge)

		# introduce a defect
		vdeleted = selection_func(choices, initial_g, past_selections)
		deletion_func(solver, vdeleted)

		past_selections.append(vdeleted)
		choices.remove(vdeleted)

#		if step > 125 and step % 3 == 0:
#			debug_cyclebasis_planar_nx(g, expected=cyclebasis)

		runtime = time.time() - t

		step_info['runtime'].append(runtime)
		step_info['current'].append(current)
		step_info['deleted'].append([vdeleted]) # list in anticipation of multiple deletions

		if verbose:
			print('step:', step, 'time:', runtime, 'current:', current)

	return trial_info

def debug_cyclebasis_planar_nx(g, expected):
	cyclebasis = cyclebasis_planar_nx(g)
	if len(cyclebasis) != len(expected):
		cbpath = 'err.cycles'
		gpath  = 'err.gpickle'
		print('cyclebasis length mismatch!')
		print('writing {} and {}'.format(cbpath, gpath))
		xs = nx.get_node_attributes(g, 'x')
		ys = nx.get_node_attributes(g, 'y')
		pos = {v:(xs[v],ys[v]) for v in g}
		write_cyclebasis(gcyclebasis, cbpath)
		write_plottable_nx(g, gpath, pos)
		import sys; sys.exit(1)

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

def cyclebasis_planar_nx(g):
	xs = {v:g.node[v]['x'] for v in g}
	ys = {v:g.node[v]['y'] for v in g}

	return planar_cycle_basis.planar_cycle_basis_nx(g, xs, ys)

def read_graph(path):
	g = nx.read_gpickle(path)
	return g

def get_deletable_nodes(g):
	return set(v for v in g if g.node[v][VATTR_REMOVABLE])

def get_measured_edge(g):
	measure_s = g.graph[GATTR_MEASURE_SOURCE]
	measure_t = g.graph[GATTR_MEASURE_TARGET]
	return (measure_s, measure_t)

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
	# FIXME using builder here is convoluted
	# FIXME in fact if you think about it this function doesn't actually
	#       accomplish anything now
	# FIXME also clarify what the EATTR/etc constants are really for
	builder = CircuitBuilder(g)
	for (v1,v2) in g.edges():
			eattr = g.edge[v1][v2]
			s = eattr[EATTR_SOURCE]
			t = other_endpoint_nx((v1,v2), s)

			builder.make_component(s, t,
				resistance = eattr[EATTR_RESISTANCE],
				voltage = eattr[EATTR_VOLTAGE],
			)
	return builder.build()

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
	return [circuit.arbitrary_edge(u,v) for (u,v) in vpath.edges(vertex_path)]

def copy_without_attributes_nx(g):
	result = nx.Graph()
	result.add_nodes_from(g)
	result.add_edges_from(g.edges())
	return result

def set_node_attributes_checked(g, name, d):
	if any(v not in d for v in g):
		raise KeyError('attribute dict does not include all nodes!')
	nx.set_node_attributes(g, name, d)

def write_plottable_nx(g, path, pos):
	outg = copy_without_attributes_nx(g)
	set_node_attributes_checked(outg, 'pos', pos)

	nx.write_gpickle(outg, path)

def write_cyclebasis(cyclebasis, path):
	s = json.dumps(cyclebasis)
	with open(path, 'w') as f:
		f.write(s)

if __name__ == '__main__':
	main()
