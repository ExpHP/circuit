
if __name__ == '__main__' and __package__ is None:
	__package__ = 'defect.plotting'

import sys
from functools import partial
from defect.circuit import load_circuit
from defect import analysis
import defect.filetypes.internal as fileio
import networkx as nx

EATTR_CHANGE = 'delta'     # float - change in current
EATTR_SOURCE = 'src'       # node - positive-sign source for EATTR_CHANGE
EATTRS = [EATTR_CHANGE, EATTR_SOURCE]
VATTR_POS    = 'pos'       # (x,y) - position (for display)
VATTR_DEFECT = 'newdefect' # bool - was defect introduced between steps?
VATTRS = [VATTR_POS, VATTR_DEFECT]

def main():
	import argparse
	parser = argparse.ArgumentParser()
	add_modes(parser)
	args = parser.parse_args()

	if hasattr(args, 'func'): # workaround for Python 3.3+: http://bugs.python.org/issue16308
		args.func(args)
	else:
		parser.print_usage(file=sys.stderr)
		print('Error: No mode specified', file=sys.stderr)
		sys.exit(1)

def add_modes(parser):
	subparsers = parser.add_subparsers()
	add_generate_mode(subparsers, 'create')
	add_plot_mode(subparsers, 'plot')

def add_generate_mode(subparsers, cmd):
	sub = subparsers.add_parser(cmd, description='generate plot data')
	sub.add_argument('circuit', type=str, help='path to initial circuit')
	sub.add_argument('cyclebasis', type=str, help='path to initial cycles')
	sub.add_argument('gpos', type=str, help='path to gpos file')
	sub.add_argument('results', type=str, help='path to results.json from trial')
	sub.add_argument('trialid', type=int, help='trial index in results.json')
	sub.add_argument('step', type=int, help='trial step for which to compute currents')
	def callback(args):
		data = Data.from_trial_data(
			circuit = load_circuit(args.circuit),
			cycles = fileio.cycles.read_cycles(args.cyclebasis, repeatfirst=True),
			pos = fileio.gpos.read_gpos(args.gpos),
			resultinfo = analysis.read_info(args.results),
			trialid = args.trialid,
			step = args.step,
		)
		# FIXME -- TERRIBLE HACK for obvious reasons
		outpath = args.results[:-len('.results.json')]
		outpath += '-{}-{}-currents.dat'.format(args.trialid, args.step)

		print("Writing to '{}'...".format(outpath))
		data.save(outpath)

	sub.set_defaults(func = callback)

def add_plot_mode(subparsers, cmd):
	sub = subparsers.add_parser(cmd, description='display plot data')
	sub.add_argument('data', type=str, help='path to -currents.dat')
	sub.add_argument('--output', '-o', type=str, default='out.pdf', help='path to plot')
	sub.add_argument('--noshow', dest='show', action='store_false', help='no preview')
	def callback(args):
		data = Data.load(args.data)
		plot_data(data, args.output, args.show)

	sub.set_defaults(func = callback)


class Data:
	def __init__(self, g):
		self.g = g

	@classmethod
	def from_trial_data(cls, circuit, cycles, pos, resultinfo, trialid, step):
		# several parts of this script would require reconsideration to support trials
		# where the vertices weren't actually deleted
		if resultinfo['defect_mode']['mode'] != 'direct removal':
			raise RuntimeError('This script was only written with "remove" mode in mind')

		# Find change in currents
		before = analysis.trial_edge_currents_at_step(circuit, cycles, resultinfo, trialid, step)
		after  = analysis.trial_edge_currents_at_step(circuit, cycles, resultinfo, trialid, step+1)

		before = edict_remove_redundant_entries(before)
		delta  = {e: after.get(e, 0.0) - before[e] for e in before}

		# NOTE: this reconstruction will lack some isolated vertices
		defects = set(get_added_defects(resultinfo, trialid, step))
		defect_dict = {v:False for v in edict_vertices(delta)}
		defect_dict.update({v:True for v in defects})

		sources = {(s,t):s for (s,t) in delta}

		g = nx.Graph()
		g.add_edges_from(delta)
		g.add_nodes_from(defects) # in case any of the defects are isolated vertices

		# ``pos`` contains all vertices from the graph's initial state.  Limit it to those currently
		#  in the modified graph (because set_node_attributes crashes on nodes that don't exist)
		pos = {k:v for k,v in pos.items() if k in g}

		nx.set_edge_attributes(g, EATTR_CHANGE, delta)
		nx.set_edge_attributes(g, EATTR_SOURCE, sources)
		nx.set_node_attributes(g, VATTR_POS, pos)
		nx.set_node_attributes(g, VATTR_DEFECT, defect_dict)
		return cls(g)

	def save(self, path):
		fileio.graph.write_networkx(self.g, path)

	@classmethod
	def load(cls, path):
		g = fileio.graph.read_networkx(path)
		for attr in EATTRS:
			if len(nx.get_edge_attributes(g, attr)) == 0:
				raise RuntimeError('Graph is missing edge attribute {}.'.format(repr(attr)))
		for attr in VATTRS:
			if len(nx.get_node_attributes(g, attr)) == 0:
				raise RuntimeError('Graph is missing node attribute {}.'.format(repr(attr)))
		return cls(g)

# for a dict assigning attributes to an undirected graph, ensure each edge only appears once
def edict_remove_redundant_entries(d):
	d = dict(d)
	for (s,t) in list(d):
		if (s,t) in d and (t,s) in d:
			del d[(t,s)]
	return d

# flatten iterable
def flat(it):
	for x in it: yield from x

# get all vertices that appear in an edge dict
def edict_vertices(d):
	return set(flat(d))

# vertices deleted between step and step+1
def get_added_defects(resultinfo, trialid, step):
	trials = resultinfo['trials']
	trial = trials[trialid]
	arr = analysis.trial_array(trial)
	return arr['deleted'][step+1]

def remove_nodes(it, removed):
	return set(it) - set(removed)

def remove_edges(it, removed):
	it = set(it)
	for s,t in removed:
		it.discard((s,t))
		it.discard((t,s))
	return it

def plot_data(data, path, show):
	import matplotlib.pyplot as plt

	# position attributes
	g = data.g
	pos = nx.get_node_attributes(g, VATTR_POS)
	delta = nx.get_edge_attributes(g, EATTR_CHANGE)

	deleted_dict = nx.get_node_attributes(g, VATTR_DEFECT)

	# things to be used as nodelists and edgelists in draw_networkx
	# (use these when defining e.g. edge_color to ensure order is consistent)
	deleted_nodes = [v for v in g if deleted_dict[v]]
	deleted_edges = [e for e in delta if any(v in deleted_nodes for v in e)] # edges attached to deleted nodes
	remaining_nodes = list(remove_nodes(g, deleted_nodes))
	remaining_edges = list(remove_edges(delta, deleted_edges))
	
	draw_g = partial(nx.draw_networkx, g, pos=pos, with_labels=False)
	draw_remaining_nodes = partial(draw_g, edgelist=[], nodelist=remaining_nodes)
	draw_deleted_nodes   = partial(draw_g, edgelist=[], nodelist=deleted_nodes)
	draw_remaining_edges = partial(draw_g, nodelist=[], edgelist=remaining_edges)
	draw_deleted_edges   = partial(draw_g, nodelist=[], edgelist=deleted_edges)

	def edge_intensities(es):
		deltas = list(map(delta.__getitem__, es))
		deltas = list(map(abs, deltas))
		delta_max = max(deltas)
		return [max(x / delta_max * 5., 0.25) for x in deltas]

	fig, ax = plt.subplots()
	ax.set_xticks([])
	ax.set_yticks([])
	draw_remaining_edges(ax=ax,
#		edge_color = [abs(ecurrents[0][e] - ecurrents[1][e]) for e in remaining_edges],
#		edge_cmap = mpl.cm.get_cmap('gist_heat_r'),
		width=edge_intensities(remaining_edges),
	)
	draw_deleted_nodes(ax=ax, node_color='g', node_size=40)
	draw_deleted_edges(ax=ax, edge_color='r', width=5.)
#	draw_deleted_edges(ax=ax, edge_color='r', width=list(range(len(deleted_edges))))
	fig.savefig(path)
	if show:
		plt.show()

if __name__ == '__main__':
	main()
