#!/usr/bin/env python3

import copy
import bisect

from defect.util import zip_variadic, zip_matching_length

import numpy as np

# FIXME this file currently simultaneously tries to support multiple paradigms
# (a bunch of free functions operating on dicts, versus gratituous use of structured numpy arrays)
#
# As a result, some properties of the data can be computed via two completely independent code paths.
# This is the *bad* kind of redundancy (the kind which can easily become inconsistent!)

def read_info(path):
	import json
	with open(path) as f:
		s = f.read()
	return json.loads(s)

def trial_defects_stepwise(trial_info):
	return map(len, trial_info['steps']['deleted'])

def trial_defects_cumulative(trial_info):
	return prefix_sums(trial_defects_stepwise(trial_info), zero=0)

def trial_max_defects_possible(trial_info):
	# FIXME this assumes there's exactly 2 non-deletable vertices
	return trial_info['graph']['num_vertices'] - 2

def trial_defects_ratio(trial_info):
	n = trial_max_defects_possible(trial_info)
	return [float(x) / n for x in trial_defects_stepwise(trial_info)]

def trial_current(trial_info):
	return trial_info['steps']['current']

def trial_resistance(trial_info):
	return map(lambda x: 1./x, trial_current(trial_info))

def trial_array(trial_info):
	steps = trial_info['steps']
	data_iter = zip_matching_length(*steps.values())

	# numpy is REALLY picky about container types
	data_tuples = list(map(tuple, data_iter))

	# autogenerate record type based on python value types
	# not sure how reliable this will be versus hardcoded names/types
	spec = []
	for name, value in zip(steps.keys(), data_tuples[0]):
		spec.append((name, type(value)))

	assert all(type(v) == typ for v,(name,typ) in zip(data_tuples[0], spec))
	return np.array(data_tuples, dtype=spec)

def trialset_array(trial_infos):
	return np.vstack(list(trial_array(info) for info in trial_infos))

# FIXME there's some redundant functionality between this and previously existing "trialset"
#  functions currently as the two different interfaces kind of work at odds to eachother
def trialset_augmented_array(trialset_info):
	arr = trialset_array(trialset_info)

	fields = listify_dtype(arr.dtype)
	oldnames = [x[0] for x in fields]
	fields.append(('step', 'i8'))
	fields.append(('deleted_cum', 'O'))
	fields.append(('deleted_count', 'i8'))
	fields.append(('deleted_count_cum', 'i'))
	fields.append(('deleted_ratio', 'f8'))

	out = np.zeros(arr.shape, dtype=fields)
	for name in oldnames:
		out[name] = arr[name]

	for t_out,t_info in zip(out, trialset_info):
		t_out['step'] = np.arange(len(t_out))
		t_out['deleted_cum'] = trial_deleted_cum_array(t_info)

	out['deleted_count'] = np.vectorize(len)(out['deleted'])
	out['deleted_count_cum'] = np.vectorize(len)(out['deleted_cum'])

	out['deleted_ratio'] = np.float64(out['deleted_count_cum']) / trial_max_defects_possible(trialset_info[0])
	assert (out['deleted_count_cum'] == np.vectorize(len)(out['deleted_cum'])).all()
	return out

# converts a np.dtype into a python list in the format accepted by np.array
def listify_dtype(dtype):
	# np.dtype is not iterable o_O
	types = [dtype[i] for i in range(len(dtype))]
	return list(zip(dtype.names, types))

# Provides a complete list of all vertices deleted up to a point for any given step.
# To avoid O(N^2) memory requirements, each element is actually a slice of a shared master array.
def trial_deleted_cum_array(trial_info):
	arr = trial_array(trial_info)

	assert isinstance(arr['deleted'][0], list)
	pieces = [np.array(x, dtype='O') for x in arr['deleted']]
	master = np.hstack(pieces)

	lengths = np.array([len(x) for x in pieces])
	stops   = np.cumsum(lengths)
	views   = [master[:x] for x in stops]

	return object_array(views, ndim=1)

# Force creation of an object-type numpy array (``dtype='O'``) with the specified number of
#  dimensions, rather than letting numpy wing it with its crazy value-based magicks:
#
# >>> np.array([[1,2],[3,4],[5,6],[7]], dtype='O').shape    # 1d array of lists
# (4,)
# >>> np.array([[1,2],[3,4],[5,6],[7,8]], dtype='O').shape  # 2d array of integers  (!!!)
# (4, 2)
def object_array(objects, ndim=1):
	# listify objects in case it is iterable.
	# keep in mind ``np.array`` cannot be trusted with this as it auto-detects ndim
	#   and may coerce the actual objects themselves into array if they happen to be
	#   list-like and of equal length
	objects = ndim_list_create(objects, ndim)
	shape = ndim_list_shape(objects, ndim)

	out = np.zeros(shape, dtype='O')
	out[...] = objects
	return out

# create an "ndim_list" (a multidimensional list of known depth) from any iterable.
# lengths are assumed to be equal along each axis.
def ndim_list_create(src, ndim):
	assert ndim > 0
	if ndim == 1:
		return list(src)
	else:
		inner = lambda x: ndim_list_create(x, ndim-1)
		return list(map(inner, src))

def ndim_list_shape(lst, ndim):
	assert ndim > 0
	shape = [None]*ndim
	for i in range(ndim):
		shape[i] = len(lst)
		lst = lst[0]
	assert None not in shape
	return tuple(shape)

# Note: because omitting steps from the beginning/middle of the trial may mess
#  with "cumulative" properties, it's not a very good idea to use this for any
#  purpose other than cutting stuff off from the end.
def slice_steps(step_info, *args):
	sl = slice(*args)
	return {k:v[sl] for k,v in step_info.items()}

# ignores shorter trials once they are zero
def trialset_average_current(trial_infos):
	assert are_lists_consistent(trial_defects_cumulative(x) for x in trial_infos)

	currents = [trial_current(x) for x in trial_infos]

	return map(average, zip_variadic(*currents))

def trialset_defects_cumulative(trial_infos):
	return reduce_consistent_lists(trial_defects_cumulative(x) for x in trial_infos)

def trim_trial_by_current(trial_info, threshold=0.0):
	current = trial_current(trial_info)

	# current monotonically decreases --> this monotonically increases
	arr = [-x for x in current]
	val = -threshold

	zero_idx = bisect.bisect_left(arr, val)

	assert all(abs(x) > threshold*(1 - 1e-14) for x in arr[:zero_idx])
	assert all(abs(x) <= threshold*(1 + 1e-14) for x in arr[zero_idx:])

	result = copy.deepcopy(trial_info)
	result['steps'] = slice_steps(result['steps'], 0, zero_idx)
	return result

# FIXME annoying signature; unlike other "trial" functions, this takes the
#  complete info and a trial index.  This is because the deletion mode is
#  not present in the individual trial infos (can we change that?)
# FIXME also don't like the idea/placement of this in general; it basically
#  tries to simulate a trial, which means any changes to the trial runner may
#  need to be reflected here
def trial_edge_currents_at_step(g, cycles, info, trialid, step):
	from defect.trial import node_deletion
	from defect.trial.cyclebasis_provider import builder_cbupdater
	from defect.circuit import MeshCurrentSolver

	# get the deletion func
	deletion_mode = node_deletion.from_info(info['defect_mode'])
	deletion_func = deletion_mode.deletion_func

	# gather all vertices deleted at the specified step
	arr = trialset_augmented_array(info['trials'])
	deleted = arr['deleted_cum'][trialid][step]

	solver = MeshCurrentSolver(g, cycles, builder_cbupdater())
	for v in deleted:
		deletion_func(solver, v)

	currents = solver.get_all_currents()
	return currents

# True if all provided lists contain the same values up to where each is defined
# (the lists may be of different length)
def are_lists_consistent(its):
	return all(map(all_equal, zip_variadic(*its)))

# are all elements equal?
def all_equal(vals):
	vals = list(vals)
	if len(vals) == 0:
		return True

	return all(x == vals[0] for x in vals)

# takes consistent lists (lists which have the same values but possibly different
# lengths) and returns a copy of the longest list.
def reduce_consistent_lists(its):
	its = list(its)
	if len(its) == 0:
		return []

	its = [list(x) for x in its]
	if are_lists_consistent(its):
		lengths = list(map(len, its))
		return its[lengths.index(max(lengths))]
	else:
		raise ValueError('Lists are not consistent!')

def average(xs, zero=0.0):
	xs = list(xs)
	return sum(xs, 0.0)/len(xs)

def prefix_sums(xs, zero=0.0):
	sums = []
	s = zero
	for x in xs:
		s += x
		sums.append(s)
	return sums

# tests I'm too lazy to give a proper place >_>
assert all_equal([])
assert all_equal([2.])
assert all_equal([2.,2.,2.,2.])
assert not all_equal([2.,2.,1.,2.])

assert reduce_consistent_lists([]) == []
assert reduce_consistent_lists([[5.,6.],[5.,6.,7.],[5.]]) == [5.,6.,7.]

try: reduce_consistent_lists([[5.,3.],[5.,6.,7.],[5.]])
except ValueError: pass
else: assert False

