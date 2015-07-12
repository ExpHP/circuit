
import sys
import math
import random
import time
import functools
try:
	import cProfile as profile
except ImportError:
	import profile

import networkx as nx
import numpy as np
import json
import toml

from multiprocessing import Pool
from util import multiprocessing_dill, TempfileWrapper

from circuit import MeshCurrentSolver, CircuitBuilder, load_circuit
import graph.cyclebasis.planar
from resistances_common import *

from components import node_selection
from components import node_deletion
from components import cyclebasis_provider

import graph.path as vpath

import util

# TODO: maybe implement subparsers for these, and put in the node_selection/deletion
#   modules since these need to be updated for each new mode
SELECTION_MODES = {
	'uniform':  node_selection.uniform(),
	'bigholes': node_selection.by_deleted_neighbors([1,10**3,10**4,10**7]),
}
DELETION_MODES = {
	'remove':   node_deletion.annihilation(),
	'multiply': node_deletion.multiply_resistance(1000., idempotent=False),
	'assign':   node_deletion.multiply_resistance(1000., idempotent=True),
}

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='.gpickle file of networkx graph')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--jobs', '-j', type=int, default=1, help='number of trials to run in parallel')
	parser.add_argument('--trials', '-t', type=int, default=10, help='number of trials to do total')
	parser.add_argument('--steps', '-s', type=int, default=100, help='number of steps per trial')
	parser.add_argument('--cyclebasis', '-c', type=str, default=None, help='.cyclebasis file. If not provided, will '
		'check the input graph file for planar embedding information.')
	parser.add_argument('--substeps', '-x', type=int, default=1, help='number of defects added per step')
	parser.add_argument('--output-json', '-o', type=str, default=None, help='output file')
	parser.add_argument('--output-pstats', '-P', type=str, default=None, help='Record profiling info (implies --jobs 1)')
	parser.add_argument('--selection-mode', '-S', type=str, required=True, choices=SELECTION_MODES, help='TODO')
	parser.add_argument('--deletion-mode', '-D', type=str, required=True, choices=DELETION_MODES, help='TODO')

	args = parser.parse_args(sys.argv[1:])

	if (args.output_pstats is not None and args.jobs != 1):
		print('error: --output-pstats/-P is limited to --jobs 1',file=sys.stderr)
		print('In other words: No multiprocess profiling!',file=sys.stderr)
		sys.exit(1)

	if (args.substeps < 1):
		print('--substeps must be a positive integer.',file=sys.stderr)
		sys.exit(1)

	# save the user some grief; fail early if output paths are not writable
	for path in (args.output_json, args.output_pstats):
		if path is not None:
			error_if_not_writable(path)

	selector = SELECTION_MODES[args.selection_mode]
	deletor = DELETION_MODES[args.deletion_mode]

	if args.cyclebasis is not None:
		cbprovider = cyclebasis_provider.from_file(args.cyclebasis)
	else:
		cbprovider = cyclebasis_provider.planar()

	# FIXME  name/return value discrepancy
	g, config = read_graph(args.input)

	# The function that worker threads will invoke
	cmd_once = functools.partial(run_trial_nx,
		# g is deliberately missing from this invocation
		steps = args.steps,
		substeps = args.substeps,
		cbprovider = cbprovider,
		selection_func = selector.selection_func,
		deletion_func  = deletor.deletion_func,
		measured_edge = config.get_measured_edge(),
		no_defect = config.get_no_defect(),
		verbose = args.verbose,
	)
	# Pass g via a temp file in case it is extremely large.
	cmd_wrapper = TempfileWrapper(cmd_once, g=g) # must keep a reference to this
	cmd_once = cmd_wrapper.func

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
	info['cyclebasis_gen'] = cbprovider.info()

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
	return multiprocessing_dill.map(p, run_with_seed, seeds, chunksize=1)

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

def run_trial_nx(g, steps, cbprovider, selection_func, deletion_func, measured_edge, *, no_defect=[], substeps=1, verbose=False):
	no_defect = set(no_defect)

	if verbose:
		print('Starting')

	initial_g = g.copy()

	trial_info = {
		'graph': graph_info_nx(g),
		'steps': {'runtime':[], 'current':[], 'deleted':[]},
	}

	is_deletable = lambda v: (v not in no_defect) and (v not in measured_edge)
	choices = [v for v in g if is_deletable(v)]
	solver  = MeshCurrentSolver(g, cbprovider.new_cyclebasis(g), cbupdater=cbprovider.cbupdater())

	past_selections = []

	step_info = trial_info['steps']
	for step in range(steps):
		t = time.time()

		# introduce defects
		deleted = []
		if step > 0:  # first step is initial state

			step_substeps = min(substeps, len(choices))
			if step_substeps == 0:
				break  # graph can't change from previous step;  end trial

			for _ in range(step_substeps):

				vdeleted = selection_func(choices, initial_g, past_selections)
				deletion_func(solver, vdeleted)

				past_selections.append(vdeleted)
				choices.remove(vdeleted)

				deleted.append(vdeleted)

		# the big heavy calculation!
		current = solver.get_current(*measured_edge)

		runtime = time.time() - t

		step_info['runtime'].append(runtime)
		step_info['current'].append(current)
		step_info['deleted'].append(deleted)

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

def read_graph(path):
	# TODO allow manual specification of config path
	g = load_circuit(path)
	for ext in ('.circuit', '.gpickle'): # HACK
		if path.endswith(ext):
			cfgpath = swap_ext(path, ext, '.defect.toml')
	config = Config.from_file(cfgpath)
	return g, config

def swap_ext(path, ext1, ext2):
	assert path.endswith(ext1)
	return path[:-len(ext1)] + ext2

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

def other_endpoint_nx(e, s):
	(v1,v2) = e
	if s == v1: return v2
	elif s == v2: return v1
	else:
		raise ValueError('{} is not an endpoint of the edge ({},{})'.format(s,v1,v2))

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

class Config:
	def __init__(self, measured_edge=None, no_defect=None):
		self.__edge = None
		self.__no_defect = None
		if measured_edge is not None: self.set_measured_edge(*measured_edge)
		if no_defect is not None: self.set_no_defect(no_defect)

	def set_measured_edge(self, s, t): self.__edge = [s,t]
	def set_no_defect(self, d):     self.__no_defect = list(d)

	def get_measured_edge(self): return tuple(self.__edge)
	def get_no_defect(self):  return list(self.__no_defect)

	@classmethod
	def from_file(cls, path):
		with open(path) as f:
			s = f.read()
		return cls.deserialize(s)

	def save(self, path):
		s = self.serialize()
		with open(path, 'w') as f:
			f.write(s)

	@classmethod
	def deserialize(cls, s):
		d = toml.loads(s)
		measured_edge = tuple(d['general']['measured_edge'])
		no_defect = list(d['general']['no_defect'])
		return cls(measured_edge, no_defect)

	def serialize(self):
		d = {
			'general': {
				'measured_edge': self.__edge,
				'no_defect': self.__no_defect,
			},
		}
		return toml.dumps(d)

if __name__ == '__main__':
	main()
