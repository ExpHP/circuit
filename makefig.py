#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from argparse import ArgumentParser
from datetime import datetime
from functools import partial
import json
import math
import sys

from defect.util import zip_matching_length
from defect.util.array import smash_equal
from defect.util.fit import *
from defect import analysis

# TODO:  Make this file stop existing (aside from its ghost that
#         will forever haunt the commit history)

def main():
	parser = ArgumentParser('Plot data')
	parser.add_argument('input', nargs='+', help='.json output file from resistances.py')

	'''
	args = parser.parse_args(sys.argv[1:])

	infos = __main__get_infos(args.input)
	metas = __main__get_metadata(args.input, infos)
	'''

#	fig_linear('fig-linear', infos, metas, xscale=100.)
#	fig_loglog('fig-loglog', infos, metas, xscale=100.)
#	fig_biglinear('fig-biglinear', infos, metas, xscale=1.)
	#fig_break('fig-break')

#	create_pdfs()
#	create_pngs()
#	plt.show()

def __main__get_infos(paths):
	infos = []
	for path in paths:
		info = load_json(path)

		print()
		print('File:',path)
		print_miscellany(info)

		infos.append(info)
	assert len(infos) == len(paths)
	return infos

class Metadata:
	def __init__(self, *, name, color):
		self.name  = name
		self.color = color

def __main__get_metadata(paths, infos):
	colors = 'rbgmkc'

	result = []
	iota = range(len(paths))
	for i, path, info in zip_matching_length(iota, paths, infos):
		prefix = common_path_prefix(paths)
		name = path[len(prefix):]

		assert len(name) > 0
		assert prefix + name == path

		# XXX TERRIBLE QUICK HACK XXX
		name = {
			'final-5.json':    'R × 5',
			'final-10.json':   'R × 10',
			'final-100.json':  'R × 100',
			'final-1000.json': 'R × 1000',
			'final-remove-ext.json': 'Remove',
		}[name]

		result.append(Metadata(name=name, color=colors[i % len(colors)]))

	return result

#----------------

def do_power_law_fits(xydata, metas, ax, xlim):
	ylim = ax.get_ylim()
	for (x, ys, yavg), meta in zip(xydata, metas):
		fit = PowerLawModel.from_data(x[-200:]*100, yavg[-200:])
		px = scale_space(scale, xlim[0], xlim[1], 150)
		ax.plot(px, fit(px), ':', c='k')
		print(meta.name, '--', '{:.5f}'.format(fit))
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

#----------------

# TODO this plot needs a replacement in the data repo before I can get
#   rid of this file

# everything in these functions are extremely dependent on a specific
#  output file
def fig_break(figname):
	fig_break0(figname + '-jump', 5325//5)
	fig_break0(figname + '-nojump', 5300//5)

def fig_break0(figname, step):
	fig = make_figure(figname)
	axa = fig.add_subplot(111)

	# HACK - load specific info
	path  = 'final-remove-ext.json'
	info, = __main__get_infos([path])
	meta, = __main__get_metadata([path], [info])

	scale      = 'linear'    # matplotlib scale; 'log' or 'linear'
	resistance = True
	percent    = False
	xlim       = (5000, 5500)
	(x,ys,yavg), = get_xydata([info], xlim=xlim, resistance=resistance, percent=percent)
	setup_axis(axa, xlim=xlim, scale=scale, resistance=resistance, percent=percent)

	steps = [step, step+1]
	offsetsteps = [x - xlim[0]//5 for x in steps]
	
	axa.plot(x, ys[0], c='k')
	axa.plot(x[offsetsteps], ys[0,offsetsteps], 'or', ms=10)
	fig.tight_layout() # XXX HACK

	#----------- graph
	# HACK - need data, too
	data = analysis.trialset_augmented_array(info['trials'])
	gs = [graph_at_step(data[0], x) for x in steps]

	# position attributes
	vxs = nx.get_node_attributes(gs[0],'x')
	vys = nx.get_node_attributes(gs[0],'y')
	pos = {v:(vxs[v],vys[v]) for v in gs[0]}

	es = [undirected_edge_set(g) for g in gs]

	# difference in currents
	from defect import circuit
	ecurrents = [circuit.compute_circuit_currents(g) for g in gs]

	# leave out battery before drawing
	for g in gs:
		g.remove_nodes_from(['top', 'bot'])

	# things to be used as nodelists and edgelists in draw_networkx
	# (use these when defining e.g. edge_color to ensure order is consistent)
	deleted_edges = list(es[0] - es[1])
	deleted_nodes = list(set(gs[0]) - set(gs[1]))
	remaining_edges = list(es[1])
	remaining_nodes = list(gs[1])
	
	draw_g = partial(nx.draw_networkx, gs[0], pos=pos, with_labels=False)
	draw_remaining_nodes = partial(draw_g, edgelist=[], nodelist=remaining_nodes)
	draw_deleted_nodes   = partial(draw_g, edgelist=[], nodelist=deleted_nodes)
	draw_remaining_edges = partial(draw_g, nodelist=[], edgelist=remaining_edges)
	draw_deleted_edges   = partial(draw_g, nodelist=[], edgelist=deleted_edges)

	# current change plot
	losses = [abs(ecurrents[0][e] - ecurrents[1][e]) for e in remaining_edges]
	losses_max = max(losses)
	widths = [max(x / losses_max * 5., 0.25) for x in losses]

	fig = make_figure(figname+'-g2')
	ax = fig.add_subplot(111)
	ax.set_xticks([])
	ax.set_yticks([])
	draw_remaining_edges(ax=ax,
#		edge_color = [abs(ecurrents[0][e] - ecurrents[1][e]) for e in remaining_edges],
#		edge_cmap = mpl.cm.get_cmap('gist_heat_r'),
		width=widths,
	)
	draw_deleted_nodes(ax=ax, node_color='g', node_size=40)
	draw_deleted_edges(ax=ax, edge_color='r', width=5.)
#	draw_deleted_edges(ax=ax, edge_color='r', width=list(range(len(deleted_edges))))

#----------------

# wrapper funcs around logic shared by basically all figures

def get_xydata(infos, *, xlim, resistance, percent):
	xydata = [xy_data_from_file(o, resistance=resistance, ignore_zero=False, percent=percent) for o in infos]
	xydata = [xy_data_apply_xlim(o, xlim) for o in xydata]
	return xydata

def setup_axis(ax, *, xlim, scale, resistance, percent):
	assert isinstance(scale, str)
	assert isinstance(resistance, bool)
	assert isinstance(percent, bool)
	ax.set_xlim(*xlim)
	ax.set_xscale(scale)
	ax.set_yscale(scale)
	set_xy_labels(ax, resistance=resistance, percent=percent)

#----------------

# this is here to help ensure labels are consistent with the actual data, by inspecting
# the corresponding arguments to xy_data_from_file
def set_xy_labels(ax, *, resistance, percent):
	assert isinstance(resistance, bool)
	assert isinstance(percent, bool)
	if percent: x_is_defect_ratio(ax)
	else:       x_is_defect_count(ax)
	if resistance: y_is_resistance(ax)
	else:          y_is_current(ax)

def y_is_current(ax):
	ax.set_ylabel('Current (V0/R0)')

def y_is_resistance(ax):
	ax.set_ylabel('Resistance (R0)')

def x_is_defect_count(ax):
	ax.set_xlabel('Defect count')

def x_is_defect_ratio(ax):
	ax.set_xlabel('Defect %')

#----------------

# this function does far too many things...
def xy_data_from_file(fileinfo, *, resistance=True, ignore_zero=True, percent=True, cutoff=0.0):

	assert isinstance(resistance, bool)
	assert isinstance(ignore_zero, bool)
	assert isinstance(cutoff, float)

	data = analysis.trialset_augmented_array(fileinfo['trials'])

	currents = data['current']
	currents = np.where(currents <= cutoff, 0.0, currents)

	# Average, possibly ignoring zeros
	mask         = np.logical_and(ignore_zero, currents == 0.0)
	current_mean = np.ma.masked_array(currents, mask).mean(axis=0)

	# exclude steps where all values were ignored
	indices = list(range(len(current_mean)))
	indices = [i for i in indices if current_mean[i] is not np.ma.masked]

	if percent: xs = data['deleted_ratio']
	else:       xs = data['deleted_count_cum']
	x = smash_equal(xs, axis=0) # all trials in a file have equal defect counts, and we'd have
	                            # other problems if this were no longer the case, anyways
	                            # (i.e. how would we take an average of y values?)

	x = x[indices]
	ys = currents[:, indices]
	yavg = (current_mean.data)[indices]
	if resistance:
		ys   = 1./ys
		yavg = 1./yavg # deliberately not the same as 'average resistance'

	assert x.shape == ys[0].shape == yavg.shape
	return x, ys, yavg

def xy_data_apply_xlim(xy_data, xlim):
	xmin,xmax = xlim
	xmin -= 1e-10
	xmax += 1e-10
	assert xmin < xmax

	x, ys, yavg = xy_data
	indices = [j for j,v in enumerate(x) if xmin <= v <= xmax]
	assert len(indices) > 0
	return x[indices], ys[:,indices], yavg[indices]

#--------------------------------------------------------------------

def load_json(path):
	with open(path) as f:
		s = f.read()
	return json.loads(s)

def trialset_data(trials):
	data = analysis.trialset_array(trials)
	assert data.ndim == 2
	return data

def common_path_prefix(paths):
	import os
	# bizarrely, despite being in os.path, this doesn't necessarily return
	#  an actual path (it's character-by-character)
	prefix = os.path.commonprefix(paths)
	if os.sep in prefix:
		stop = prefix.rfind(os.sep) + len(os.sep)
		prefix = prefix[:stop]
	else:
		prefix = ''

	assert prefix.endswith(os.sep) or len(prefix) == 0
	return prefix

#--------------------------------------------------------------------

def axis_xspace(ax, n=50):
	start,stop = ax.get_xlim()
	scale = ax.get_xscale()
	return scale_space(scale, start, stop, n)

# uniform interface to linspace and logspace in which start and stop
#   are always the actual values (never the logarithm)
#
# First argument is the matplotlib scale name ('log' or 'linear')
def scale_space(scale, start, stop, n=50):
	assert isinstance(n, int)
	assert isinstance(scale, str)
	if scale == 'log':
		if start == 0.:     # FIXME I don't like that this is handled here
			start = 1.e-10  #  as there's simply no telling a reasonable default
		return np.logspace(
			math.log(start) / math.log(10.),
			math.log(stop) / math.log(10.),
			n,
		)
	elif scale == 'linear':
		return np.linspace(start, stop, n)
	assert False

#--------------------------------------------------------------------

def print_miscellany(info, file=None):

	if file is None:
		file = sys.stdout # bind at runtime

	def print_mode_info(heading, d, primary_key):
		if len(d) == 1:
			extra = ''
		else:
			extra_items = [(k,v) for (k,v) in d.items() if k != primary_key]
			extra_strs = ['{}:{}'.format(str(k),repr(v)) for k,v in extra_items]
			extra = '({})'.format(', '.join(extra_strs))
		print(heading, d[primary_key], extra, file=file)

	print_mode_info('Defect mode:', info['defect_mode'], 'mode')
	print_mode_info('Selection mode:', info['selection_mode'], 'mode')
	print('Number of trials:', len(info['trials']), file=file)
	print('Started:', format_time(info['time_started']), file=file)
	print('Duration:', format_timedelta(info['time_started'], info['time_finished']), file=file)

def format_time(timestamp):
	return datetime.utcfromtimestamp(timestamp).strftime('%a, %x %X') + " UTC"

def format_timedelta(start, end):
	return str(datetime.utcfromtimestamp(end) - datetime.utcfromtimestamp(start))

#--------------------------------------------------------------------

def graph_at_step(trialdata, step):
	assert trialdata.ndim == 1
	g = nx.read_gpickle('data/hex_100_100.gpickle')
	g.remove_nodes_from(trialdata['deleted_cum'][step])
	return g

def undirected_edge_set(g):
	return set([tuple(sorted(x)) for x in g.edges()])

#--------------------------------------------------------------------

FIGURES = []
def make_figure(basename):
	global FIGURES
	fig = plt.figure(figsize=(5,4))
#	fig = plt.figure(figsize=(2.25,2))
	FIGURES.append((basename, fig.number))
	return fig

def create_pdfs():
	for basename, number in FIGURES:
		plt.figure(number)
		plt.savefig("{}.pdf".format(basename), format='pdf')

def create_pngs():
	for basename, number in FIGURES:
		plt.figure(number)
		plt.savefig("{}.png".format(basename), format='png')

#--------------------------------------------------------------------

if __name__ == '__main__':
	main()

