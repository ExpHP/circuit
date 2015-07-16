#!/usr/bin/env python3

import time

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import defect.filetypes.internal as fileio
from defect.circuit import load_circuit, MeshCurrentSolver

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('basename', type=str)
	args = parser.parse_args()

	circuit = load_circuit(args.basename + '.circuit')
	cycles = fileio.cycles.read_cycles(args.basename + '.cycles')

	badcycles = nx.cycle_basis(circuit)
	for c in badcycles: c.append(c[0])

#	fig, ax = plt.subplots()
#	plot2(get_rmatrix(circuit, cycles), ax)
#	fig, ax = plt.subplots()
#	plot2(get_rmatrix(circuit, badcycles), ax)
#	plt.show()

#	write_file(get_rmatrix(circuit, cycles), 'good-rmatrix.dat')
#	write_file(get_rmatrix(circuit, badcycles), 'bad-rmatrix.dat')

	print('get good')
	get_rmatrix(circuit, cycles)
	print('get bad')
	get_rmatrix(circuit, badcycles)

def write_file(m, path):
	print(m.nnz)
	if m.shape[0] * m.shape[1] > 10**6:
		raise RuntimeError("don't do it! don't do it! (shape: {})".format(m.shape))
	data = m.todense()
	data = np.where(data==0., 0., 1.)
	lines = data.tolist()
	lines = [list(map(str, row)) for row in lines]
	lines = [' '.join(row) for row in lines]
	s = '\n'.join(lines)
	with open(path, 'w') as f:
		f.write(s)

def plot(m, ax=None):
	# stolen from: http://stackoverflow.com/questions/22961541
	m = m.tocoo()
	if ax is None:
		_, ax = plt.subplots()

	ax.set_axis_bgcolor('black')
	ax.plot(m.col, m.row, 's', color='white', ms=1)
	ax.set_xlim(0, m.shape[1])
	ax.set_ylim(0, m.shape[0])
	ax.set_aspect('equal')
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.invert_yaxis()
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	return ax

def plot2(m, ax=None):
	print(m.nnz)
	m = m.tocoo()
	if ax is None:
		_, ax = plt.subplots()

	if m.shape[0] * m.shape[1] > 10**6:
		raise RuntimeError("don't do it! don't do it! (shape: {})".format(m.shape))
	#ax.set_axis_bgcolor('black')
	data = m.todense()
	data = np.where(data==0., 0., 1.)
	ax.imshow(data, cmap='gray',interpolation='nearest')
#	ax.set_xlim(0, m.shape[1])
#	ax.set_ylim(0, m.shape[0])
#	ax.set_aspect('equal')
	for spine in ax.spines.values():
		spine.set_visible(False)
#	ax.invert_yaxis()
#	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	return ax

#---------------------------------------------------------

def get_rmatrix(circuit, cyclebasis):
	mcs = MeshCurrentSolver(circuit, cyclebasis)
	r = mcs.resistance_matrix.get()
	t = time.time()
	mcs.cycle_currents.get()
	print(time.time()-t)
	return r

'''
def serialize(m):
	m = m.tocsc()
	d = {}
	d['shape']   = list(m.shape)
	d['data']    = list(m.data)
	d['indices'] = list(m.indices)
	d['indptr']  = list(m.indptr)
	return json.dumps(d)

def deserialize(s):
	d = json.loads(s)
	return scipy.sparse.csc_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])

def save(m, path):
	s = serialize(m)
	with open(path, 'w') as f:
		f.write(s)

def load(path):
	with open(path) as f:
		s = f.read()
	return deserialize(s)
'''

if __name__ == '__main__':
	main()
