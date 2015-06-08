
import sys
import bisect

import graphs.vertexpath as vpath

# TODO The scipy/numpy-related functionality is not crucial to operation here;
# Maybe just for once I'll FINALLY be able to use pypy on something :P
import scipy.sparse
import numpy as np

__all__ = [
	'main',
	'CycleBasisBuilder',
	'build_cyclebasis',
	'build_cyclebasis_terminal',
	'STAGE_CYCLES',
	'STAGE_FALLBACK',
]

THOROUGH_DEFAULT = False

STAGE_PROVIDED = 10
STAGE_FALLBACK = 20

def main(argv):
	pass

# Quick "frontend" for build_cyclebasis which reports incremental progress to stdout if verbose=True.
def build_cyclebasis_terminal(cycles, fallback, thorough=THOROUGH_DEFAULT, verbose=False):
	cycles = list(cycles)
	fallback = list(fallback)
	if verbose:
		print('Generating cyclebasis of length {} from {} provided cycles'.format(len(fallback), len(cycles)))

		stage_names = {
			STAGE_PROVIDED:'provided',
			STAGE_FALLBACK:'fallback',
		}
		def callback(d):
			# numeric lengths for alignment
			len_total = len(str(d['total_needed']))
			len_stage = len(str(d['stage_length']))

			found_ratio = '{{0[total_found]:{0}d}} / {{0[total_needed]:{0}d}}'.format(len_total).format(d)
			stage_name = stage_names[d['stage_id']]
			stage_ratio = '{{0[stage_current]:{0}d}} / {{0[stage_length]:{0}d}}'.format(len_stage).format(d)

			sys.stdout.write('\r')
			sys.stdout.write('Cycles found: {} .  '.format(found_ratio))
			sys.stdout.write('Searching {} cycles (Progress: {})'.format(stage_name, stage_ratio))

			# start new line between stages
			if d['stage_current'] == d['stage_length']:
				sys.stdout.write('\n')

			sys.stdout.flush()

	else:
		def callback(d):
			pass

	return build_cyclebasis(cycles, fallback, thorough, callback)

# If thorough=False, stops checking when len(fallback) cycles have been found (thorough=True continues
# checking all the way through the end, reporting an error if another linearly independent cycle is found;
# this may indicate an incomplete fallback cyclebasis, or invalid edges in the provided cycles)
def build_cyclebasis(cycles, fallback, thorough=THOROUGH_DEFAULT, progress_callback=lambda d:None):
	cycles = list(cycles)
	fallback = list(fallback)

	cbbuilder = CycleBasisBuilder()

	def emit_progress(*, stage_id, stage_length, stage_current):
		progress_callback({
			'stage_id': stage_id,
			'stage_length': stage_length,
			'stage_current': stage_current,
			'total_found': len(cbbuilder.cycles),
			'total_needed': len(fallback),
		})

	def add_cycles_from(lst, stage):
		emit_progress(
			stage_id = stage,
			stage_length = len(lst),
			stage_current = 0,
		)
		for i,cycle in enumerate(lst):

			if cycle[0] != cycle[-1]:
				cycle.append(cycle[0])

			cbbuilder.add_if_independent(cycle)

			# this is done after checking the cycle and before exiting the loop so that
			#  the last iteration looks like 30000/30000 (as opposed to 29999/30000)
			emit_progress(
				stage_id = stage,
				stage_length = len(lst),
				stage_current = i+1,
			)

			if (not thorough) and len(cbbuilder.cycles) >= len(fallback):
				break

			if len(cbbuilder.cycles) > len(fallback):
				# Could be due to a bad fallback, or the suggested cycles might contain invalid edges;
				# Make no assumptions.
				raise RuntimeError('Found more than len(fallback) linearly independent cycles!')

	add_cycles_from(cycles, stage=STAGE_PROVIDED)
	add_cycles_from(fallback, stage=STAGE_FALLBACK)

	assert len(cbbuilder.cycles) == len(fallback)
	return cbbuilder.cycles


#----------------------

class EdgeIndexMapper:
	def __init__(self):
		self.edges = []
		self.edge_indices = {}

	def map_path(self, path):
		result = []
		for s,t in vpath.edges(path):
			if (s,t) not in self.edge_indices:
				self.edge_indices[s,t] = len(self.edges)
				self.edge_indices[t,s] = len(self.edges)
				self.edges.append((s,t))
			result.append(self.edge_indices[s,t])
		return result

# A matrix with values modulo 2, kept in rref form.
class SparseRrefBitMatrix:

	def __init__(self):
		self.rows         = [] # list of sets containing columns of each 1
		self.leading_ones = [] # column of first 1 for each row  (None for all-zero rows)
		self.nnz_rows = 0 # number of nonzero rows

	# reduces a given row (as a set of column numbers containing 1) until it does not
	#  contain any ones that conflict with a leading 1 in the matrix.
	# If prereduce_only is True, it stops when it encounters the first 1 that does not
	#  conflict. (this is enough to detect if a row is a linear combination of rows in
	#  the matrix; prereducing such a row will leave all zeros)
	def reduce_row(self, ones, prereduce_only=False):
		ones = set(ones)
		result = set()

		while len(ones) > 0:
			# locate minimum from scratch each time (rather than iterating a sorted list).
			# necessary because the XOR operation can cause new 1s to appear.
			col = min(ones)

			leader = self._leading_row(col)
			if leader is None:

				# no conflict; move to result
				ones.remove(col)
				result.add(col)

				if prereduce_only:
					break
				else:
					continue

			# resolve conflict with leading 1 row
			assert min(ones) == min(leader)
			ones ^= leader

		if prereduce_only:
			result |= ones # we stopped early, there may be some left
			assert self.row_is_prereduced(result) # postcondition
		else:
			assert len(ones) == 0
			assert self.row_is_reduced(result) # postcondition
		return result

	def row_is_prereduced(self, ones):
		return (len(ones)==0) or (self._leading_row(min(ones)) is None)

	def row_is_reduced(self, ones):
		return all(self._leading_row(col) is None for col in ones)

	# Inserts a row, maintaining reduced row echelon form.
	def insert(self, ones):
		ones = self.reduce_row(ones)

		if len(ones) == 0:
			# don't increment nnz_rows
			self.rows.append(set(ones))
			self.leading_ones.append(None)
			return

		leading = min(ones)

		i = self._insertion_index(leading)
		self.rows.insert(i, set(ones))
		self.leading_ones.insert(i, leading)
		self.nnz_rows += 1

		self._post_insertion_reduce(i)

		# doing this check every iteration has O(rows**3) total complexity;
		# a bit much even for debug
		assert self._validate_rref()

	# Row (object) with the specified leading 1, or None if there isn't one
	def _leading_row(self, leading_one):
		i = self._insertion_index(leading_one)

		assert i <= self.nnz_rows
		if (i == self.nnz_rows) or (self.leading_ones[i] != leading_one):
			return None # no such row

		return self.rows[i]

	# adds the row at rowI to any rows above it to remove any conflicts with its leading one
	def _post_insertion_reduce(self, rowI):
		leading = self.leading_ones[rowI]
		for otherI in range(rowI):
			if leading in self.rows[otherI]:
				self.rows[otherI] ^= self.rows[rowI]

	# Checks the reduced row echelon form invariant.  O(self.nnz_rows**2)
	def _validate_rref(self):
		for rowI in range(self.nnz_rows):

			leading = self.leading_ones[rowI]
			assert leading == min(self.rows[rowI])

			for otherI in range(rowI):
				assert leading not in self.rows[otherI]
		return True

	# Index where a row with a given leading one would belong
	def _insertion_index(self, leading_one):
		return bisect.bisect_left(self.leading_ones, leading_one, lo=0, hi=self.nnz_rows)

	def tocoo(self):
		rows,cols,vals = [],[],[]
		for rowI in range(self.nnz_rows):
			rows.extend([rowI]*len(self.rows[rowI]))
			cols.extend(self.rows[rowI])
			vals.extend([1]*len(self.rows[rowI]))

		trueshape = (len(self.rows), max(cols)+1)
		m = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=trueshape, dtype=int)

		assert m.max() in (0,1)
		return m

	def todense(self):
		return self.tocoo().todense()

#----------------------

# Test cases

def assertSrbmEqual(a,b):
	a = a.todense()
	b = np.array(b)
	if not np.array_equal(a, b):
		raise AssertionError('non-equal arrays:\n{}\n{}'.format(a,b))

_m1 = SparseRrefBitMatrix()
_m1.insert([1,3])
assertSrbmEqual(_m1, [[0,1,0,1]])

_m1.insert([0]) # already-reduced list which gets inserted in front
_m1.insert([4]) # already-reduced list which gets inserted at back
assertSrbmEqual(_m1, [[1,0,0,0,0],[0,1,0,1,0],[0,0,0,0,1]])

_m1.insert([2]) # already-reduced list which gets inserted in middle
assertSrbmEqual(_m1, [[1,0,0,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,1]])

_m1.insert([]) # all zeros, goes to end
assertSrbmEqual(_m1, [[1,0,0,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,0,0]])
assert _m1.nnz_rows == 4
#----------------------

_m2 = SparseRrefBitMatrix()
_m2.insert([0,1])
assertSrbmEqual(_m2, [[1,1]])

# the (binary) rows 10 and 01 will both reduce to 01
assert _m2.reduce_row([0]) == _m2.reduce_row([1]) == set([1])

_m2.insert([0])
assertSrbmEqual(_m2, [[1,0],[0,1]])

_m2.insert([1,1]) # test adding a row which is a linear combination
assertSrbmEqual(_m2, [[1,0],[0,1],[0,0]])
#----------------------

class CycleBasisBuilder:
	def __init__(self):
		self.edge_mapper = EdgeIndexMapper()
		self.bitmat = SparseRrefBitMatrix()
		self.cycles = []

	# If the provided cycle is linearly independent from the cycles already in
	#  the cyclebasis, adds the cycle and returns True.
	# Otherwise, returns False.
	def add_if_independent(self, cycle):
		if not vpath.is_cycle(cycle):
			raise ValueError('CycleBasisBuilder was provided a non-cycle')
		cycle = list(cycle)
		edgeids = self.edge_mapper.map_path(cycle)
		reduced = self.bitmat.reduce_row(edgeids, prereduce_only=True)

		if len(reduced) == 0:
			return False # linearly dependent (cycle is a sum of others)
		else:
			self.bitmat.insert(reduced)
			self.cycles.append(cycle)

			assert self.bitmat.nnz_rows == len(self.bitmat.rows)
			return True
		assert False

_cbb1 = CycleBasisBuilder()
assert _cbb1.add_if_independent('abca')
assert _cbb1.add_if_independent('dcbd')
assert not _cbb1.add_if_independent('acdba')
assert not _cbb1.add_if_independent('abdca')

if __name__ == '__main__':
	main(sys.argv)
