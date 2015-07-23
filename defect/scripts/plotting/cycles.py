#!/usr/bin/env python3

import sys
import os
from argparse import ArgumentParser

import networkx as nx
import matplotlib.pyplot as plt

import json

from defect.util import window2
from functools import partial

def main():
	# TODO
	print('This script is old and requires some TLC', file=sys.stderr)
	sys.exit(1)

	parser = ArgumentParser(sys.argv[0], description='Generate an obnoxious number of '
		'plots from a cyclebasis')
	parser.add_argument('graph', help='graph .pickle file')
	parser.add_argument('cyclebasis', help='cyclebasis .json file')
	parser.add_argument('--output-dir', '-o', required=True, help='output directory (will be created)')
	parser.add_argument('--force', '-f', action='store_true', help='overwrite existing files in directory')

	args = parser.parse_args(sys.argv[1:])

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	g = nx.read_gpickle(args.graph)
	cyclebasis = load_json(args.cyclebasis)

#	paths = [os.path.join(args.output_dir, 'cycle_{:05d}.pdf'.format(i)) for i in range(len(cyclebasis))]
	paths = [os.path.join(args.output_dir, 'cycle_{:05d}.svg'.format(i)) for i in range(len(cyclebasis))]
	for p in paths:
		if (not args.force) and os.path.exists(p):
			err('{} already exists.  Exiting.'.format(p))
			err('(no files have been written! Use -f to override.')
			sys.exit(1)
	# note race condition between this check and write_fig

	pos = nx.get_node_attributes(g, 'pos')

	for cycle,path in zip(cyclebasis, paths):
		fig = fig_cycle(g, cycle, pos)
		fig.savefig(path)
		plt.close(fig)

def err(*args):
	print(*args, file=sys.stderr)

def load_json(path):
	with open(path) as f:
		s = f.read()
	return json.loads(s)

# listify x and y to make matplotlib happy
def plot(ax, x, y, *args, **kwargs):
	ax.plot(list(x), list(y), *args, **kwargs)

def fig_cycle(g, vcycle, pos):
	fig,ax = plt.subplots()

	assert vcycle[0] == vcycle[-1]

	draw = partial(nx.draw_networkx, g, pos, with_labels=False, ax=ax)
	draw(edgelist=g.edges(), nodelist=[])
	draw(edgelist=list(window2(vcycle)), nodelist=[], edge_color='r', width=1.5)

	return fig

if __name__ == '__main__':
	main()
