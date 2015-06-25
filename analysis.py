#!/usr/bin/env python3

import copy
import bisect

from itertools import zip_longest

def trial_defects_stepwise(trial_info):
	return map(len, trial_info['steps']['deleted'])

def trial_defects_cumulative(trial_info):
	return prefix_sums(trial_defects_stepwise(trial_info), zero=0)

def trial_current(trial_info):
	return trial_info['steps']['current']

def trial_resistance(trial_info):
	return map(lambda x: 1./x, trial_current(trial_info))

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
	result['steps'] = slice_steps(result['steps'], start=0, stop=zero_idx)
	return result

# True if all provided lists contain the same values up to where each is defined
# (the lists may be of different length)
def are_lists_consistent(its):
	return all(map(all_equal, zip_variadic(*its)))

# a version of zip intended for lists of varying length, which returns tuples
#  of varying length based on how many lists remain.
def zip_variadic(*its):
	sentinel = object()
	def without_fill(xs):
		return filter(lambda x: x is not sentinel, xs)
	return (tuple(without_fill(xs)) for xs in zip_longest(*its, fillvalue=sentinel))

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

