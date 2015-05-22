
import sys
import math
import random
import time


import networkx as nx
import numpy as np

from multiprocessing import Pool
import multiprocessing_dill

from circuit import Circuit
from graphs.planar_cycle_basis import minimal_cycle_basis as planar_cycle_basis
from resistances_common import *

import cProfile

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
	#g = read_graph(args.input)

	#show_trial(200,60,steps=100,fitupto=20)
	#cProfile.run('bench_trial(70,20,steps=20)', sort='tottime')
	#cProfile.run('bench_old(40,6)',sort='tottime')
	#compare(10,5)
	#do_visualize(g)

def simple_profile(n,f,*args,**kwargs):
	for i in range(10):
		t = time.time()
		f(*args, **kwargs)
		print(i, time.time() - t)

def read_graph(path):
	g = nx.read_gpickle(path)

	return g

# A scrolling 2-element window on an iterator
def window2(it):
	it = iter(it)
	prev = next(it)
	for x in it:
		yield (prev,x)
		prev = x

def to_edge_path(g, vertex_path):
	# HACK should be SingleGraph.get_edge instead of BaseGraph.get_arbitrary_edge
	#     but I haven't implemented the former class yet
	# or better yet just get rid of my silly pointless incomplete graph library >_>
	return [g.arbitrary_edge(u,v) for (u,v) in window2(vertex_path)]

def flat_map(f, lst):
	for item in lst:
		yield from f(lst)

def other_endpoint(e, s):
	(v1,v2) = e
	if s == v1: return v2
	elif s == v2: return v1
	else:
		raise ValueError('{} is not an endpoint of the edge ({},{})'.format(s,v1,v2))

def circuit_from_nx(g):
	circuit = Circuit()
	circuit.add_vertices(iter(g))
	for (v1,v2) in g.edges():
			eattr = g.edge[v1][v2]
			s = eattr[EATTR_SOURCE]
			t = other_endpoint((v1,v2), s)

			circuit.add_component(s, t,
				resistance = eattr[EATTR_RESISTANCE],
				voltage = eattr[EATTR_VOLTAGE],
			)
	return circuit

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

	deletable = set(v for v in g if g.node[v][VATTR_REMOVABLE])

	step_current = []

	for step in range(steps):
		t = time.time()

		step_current.append(compute_current_planar_nx(g))

		r = random.choice(list(deletable))
		deletable.remove(r)
		g.remove_node(r)

		if verbose:
			print('step: ', step, 'time: ', time.time() - t)

	return step_current

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

#nx.draw_networkx(g, pos={v:(xs[v],ys[v]) for v in g})
#plt.show()

if __name__ == '__main__':
	main()
