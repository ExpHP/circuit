
import sys
import math
import random

import networkx as nx

from circuit import Circuit
from graphs.planar_cycle_basis import minimal_cycle_basis as planar_cycle_basis

import numpy as np

from resistances_common import *

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='.gml or .gml.gz file')
	args = parser.parse_args(sys.argv[1:])

	print('Reading {}'.format(args.input))
	g = read_graph(args.input)
	print('Done reading')

	import cProfile
	#show_trial(200,60,steps=100,fitupto=20)
	#cProfile.run('bench_trial(70,20,steps=20)', sort='tottime')
	#cProfile.run('bench_old(40,6)',sort='tottime')
	cProfile.runctx('bench_new(g)', globals(), locals(), sort='cumtime')
#	bench_new(g)
	#compare(10,5)
	#do_visualize(g)

def read_graph(path):
	g = nx.read_gml(path)

	# gml-stored graphs use integer labels; retrieve the original labels
	labelmap = {v:g.node[v][VATTR_LABEL] for v in g}
	g = nx.relabel_nodes(g, labelmap)

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
	for v1 in g:
		for v2,eattr in g.edge[v1].items():
			s = eattr[EATTR_SOURCE]
			t = other_endpoint((v1,v2), s)

			circuit.add_component(s, t,
				resistance = eattr[EATTR_RESISTANCE],
				voltage = eattr[EATTR_VOLTAGE],
			)
	return circuit

def trial_inputs(g):
	circuit = circuit_from_nx(g)

	xs = {v:g.node[v]['x'] for v in g}
	ys = {v:g.node[v]['y'] for v in g}

	deletable = set(v for v in g if g.node[v][VATTR_REMOVABLE])

	measure_s = g.graph[GATTR_MEASURE_SOURCE]
	measure_t = g.graph[GATTR_MEASURE_TARGET]

	assert measure_s not in deletable
	assert measure_t not in deletable

	measure_e = circuit.arbitrary_edge(measure_s, measure_t)

	return circuit, measure_e, deletable, xs, ys

def run_trial(circuit, measured_edge, deletable_vs, xs, ys):

	step_currents = []
	for stepnum in range(steps):

		# take advantage of planar representation to get
		#  a MINIMUM cycle basis
		es = {e: circuit.edge_endpoints(e) for e in circuit.edges()}
		vertex_cyclebasis = planar_cycle_basis(circuit.vertices(), es, xs, ys)

		edge_cyclebasis = [to_edge_path(circuit, cycle) for cycle in vertex_cyclebasis]

		currents = circuit.compute_currents(edge_cyclebasis)

		step_currents.append(currents[measured_edge])

#		r = g.random_vertex()
#		while r in (Vertex.TOP_CONNECTOR, Vertex.BOT_CONNECTOR):
#			r = g.random_vertex()
		r = random.choice(list(deletable_vs))
		deletable_vs.remove(r)
		circuit.delete_vertices([r])

		# hack
		del xs[r]
		del ys[r]

#		g.delete_vertices([r])
	return step_currents

def show_trial(nrows, ncols, steps, fitupto):

	x = range(steps)

	fig,ax = plt.subplots()
	ys = [run_trial(nrows, ncols, steps) for _ in range(10)]

	# get resistances
	#for i,y in enumerate(ys):
	#	for j in range(len(y)):
	#		try:
	#			y[j] = 1./y[j]
	#		except Exception:
	#			break
	#	ys[i] = y

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

def compare(g):
	circuit, measured_edge, deletable_vs, xs, ys = trial_inputs(g)

	es = {e: circuit.edge_endpoints(e) for e in circuit.edges()}
	vertex_cyclebasis = planar_cycle_basis(circuit.vertices(), es, xs, ys)
	edge_cyclebasis = [to_edge_path(circuit, cycle) for cycle in vertex_cyclebasis]

	currents_old = circuit.compute_currents(circuit.cyclebasis())
	currents_new = circuit.compute_currents(edge_cyclebasis)

	print('old',currents_old[measure_edge])
	print('new',currents_new[measure_edge])

def bench_new(g):
	circuit, measured_edge, deletable_vs, xs, ys = trial_inputs(g)

	es = {e: circuit.edge_endpoints(e) for e in circuit.edges()}
	vertex_cyclebasis = planar_cycle_basis(circuit.vertices(), es, xs, ys)
	edge_cyclebasis = [to_edge_path(circuit, cycle) for cycle in vertex_cyclebasis]

	print('basis size: {}'.format(len(edge_cyclebasis)))
	print('basis edges: {}'.format(sum(len(p) for p in edge_cyclebasis)))

	circuit.compute_currents(edge_cyclebasis)

def bench_old(g):
	circuit, measured_edge, deletable_vs, xs, ys = trial_inputs(g)

	cyclebasis = g.cycle_basis()

	print('basis size: {}'.format(len(cyclebasis)))
	print('basis edges: {}'.format(sum(len(p) for p in cyclebasis)))

	circuit.compute_currents(cyclebasis=cyclebasis)

#def do_visualize(g, vxs, vys):

	#xs = np.array([vxs[v] for v in g.vertices()])
	#ys = np.array([vys[v] for v in g.vertices()])

	#lines = []
	#for e in g.edges():
		#u,v = g.edge_endpoints(e)
		#lines.append(((vxs[u],vys[u]), (vxs[v],vys[v])))
	#lc = mc.LineCollection(lines, colors='k',linewidths=2)

	#fig,ax = plt.subplots()
	#ax.set_aspect('equal')
	#ax.scatter(xs,ys,s=25.)
	#ax.add_collection(lc)
	#plt.show()

if __name__ == '__main__':
	main()
