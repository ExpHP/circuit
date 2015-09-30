
from bisect import bisect_left, bisect_right
import unittest
import itertools
import sys

from defect.trial.node_selection import *

from scipy.stats import binom
import numpy as np
import networkx as nx

# Note:  This test module uses code auto-generation.
# With the acknowledgement that there is **almost always** a better solution than codegen,
# lemme explain.

# To test that a SelectionMode gives results according to its expected probability
# distribution, we sample its output a bunch of times and test that each result has
# a number of occurrences within some "reasonable range".  This "reasonable range"
# is a complicated function of a number of things: The occurence rate of that result,
# the total number of samples, and how frequently we will allow for a false positive.

# It would suck if it went unnoticed that changing one of these factors in some way suddenly
# made the test worthless (large ranges or too much overlap), and it's hard to guarantee
# that won't happen if the "reasonable ranges" are simply computed during the test. By
# using codegen as an intermediate step, we get a chance to inspect the ranges used by
# the test.

# The alternatives are to either allow a worthless test to go undetected, or to attempt
# to devise some sort of heuristic "metatest" for spotting one.


def weighted_prob(weights, counts, kind):
	# Get the chance of choosing a specific item of a particular ``kind`` given
	#   a weight distribution.
	# weights - list of weights for each kind
	# counts  - number of each kind
	# kind    - which kind is the item we're choosing
	assert len(weights) == len(counts)
	return weights[kind] / sum([w*c for w,c in zip(weights, counts)])


# Tests which run a SelectionMode a bunch of times and checks the distribution
#  of possible "orderings" it gives.
class DistributionTests(unittest.TestCase):
	# ======================================
	# HERE BEGINS A BUNCH OF UGLY MACHINERY

	# some status flags to keep track of where we are in the test, because the only
	# thing worse than an overly complicated unit test is an overly complicated unit
	# test that doesn't actually run!
	STATUS_PREPARE = object()
	STATUS_COLLECT = object()
	STATUS_VALIDATE = object()

	# automatically called at test beginning
	def setUp(self):
		self.status = self.STATUS_PREPARE

	def prepare(self, g, s_mode, nsamples):
		# Setup variables
		# g:         The graph to test on.
		# s_mode:    SelectionMode
		# nsamples:  How many samples will be pulled from the SelectionMethod's distribution?
		self.assertIs(self.status, self.STATUS_PREPARE, 'unit test is doing steps out of order')

		self.g = g
		self.s_mode = s_mode
		self.nsamples = nsamples

		self.status = self.STATUS_COLLECT

	def collect(self):
		# Sample the distribution of the SelectionMode
		self.assertIs(self.status, self.STATUS_COLLECT, 'unit test is doing steps out of order')

		counts = {k:0 for k in itertools.permutations(self.g.nodes())}
		for _ in range(self.nsamples):
			selector = self.s_mode.selector(self.g)
			choices = set(self.g.nodes())
			order = []
			while len(choices) > 0:
				order.append(selector.select_one(choices))
				choices.remove(order[-1])
			counts[tuple(order)] += 1

		self.counts = counts
		self.status = self.STATUS_VALIDATE

	def validate_range(self, limits, *orders):
		self.assertIs(self.status, self.STATUS_VALIDATE, 'unit test is doing steps out of order')
		lo, hi = limits
		for order in orders:
			x = self.counts.pop(tuple(order))
			if not lo <= x <= hi:
				raise AssertionError('Assertion failed: {} not in range [{}, {}]'.format(x, lo, hi))

	# automatically called at test end
	# use this opportunity to confirm that our test didn't miss anything
	def tearDown(self):
		if self._test_has_failed():
			return # one error is enough

		self.assertIs(self.status, self.STATUS_VALIDATE, 'unit test collected no data')

		# require all nonzero counts to have been tested
		# (counts are removed from the dict when tested)
		for k,v in self.counts.items():
			self.assertEqual(v, 0, 'Ordering {} has nonzero count but was not tested!'.format(k))

	def _test_has_failed(self):
		# Terrible hack to determine if the test was a failure.
		# Specific to Python 3.4+
		for method, error in self._outcome.errors:
			if error:
				return True
		return False

	# Codegen.
	# Yes, really.
	# Don't give me that look, I'm serious!
	# See comment at top of this module for justification.
	def codegen_range_checks(self, probs, failchance):
		# probs:  List of tuples of (probability, orders)
		#   where probability: float in range [0., 1.]
		#   and orders: list of strings representing selection orders with this probability.
		#
		# failchance:  Chance of Type I error (i.e. chance of a test failing for working code
		#   due to random chance alone).  Too small of a magnitude results in worthless tests
		#   unless the number of samples is very large to compensate.
		#
		# Output:
		#   Prints code for verifying that the number of samples counted for each selection order
		#   falls within acceptible bounds.
		self.assertIsNot(self.status, self.STATUS_PREPARE, 'codegen should only be done after self.prepare()')

		self.assertAlmostEqual(1., sum(sorted(p*len(os) for (p,os) in probs)), 'total probability should be 1')

		allorders = []
		for p,os in probs: allorders.extend(os)
		self.assertEqual(len(allorders), len(set(allorders)), 'all orders in probs should be unique')

		# find acceptible ranges for each item
		N = self.nsamples
		ranges = []
		for prob, orders in probs:
			cdfvals = binom.cdf(np.arange(N+1), N, prob)
			sfvals  = binom.sf(N-np.arange(N+1), N, prob) # in backwards (increasing) order for bisect

			ranges.append((
				bisect_left(cdfvals, failchance),     # lo
				N - bisect_right(sfvals, failchance)  # hi
			))

		# format string for range (align the numbers)
		lolen = max(len(str(lo)) for (lo,hi) in ranges)
		hilen = max(len(str(hi)) for (lo,hi) in ranges)
		range_fmt = '({:%dd}, {:%dd})' % (lolen, hilen)

		# sort descendingly by hi for easier comparison of ranges
		zipped = [(p,o,l,h) for (p,o),(l,h) in zip(probs,ranges)]
		zipped.sort(key=lambda tup: -tup[3])
		probs,ranges = zip(*[((p,o),(l,h)) for p,o,l,h in zipped])

		# go go gadget obnoxiously large comment
		print('######################################################')
		print('# BEGIN CODE AUTOGENERATED BY codegen_range_checks()')
		print('# Parameters:')
		print('#    Number of samples: {:d}'.format(self.nsamples))
		print('#    Chance of spontaneous failure: ~{:g}'.format(failchance))
		print('#')
		print('# The first two numbers of each line are the "range" of accepted counts in the')
		print('#  final distribution. Ideally you want to MINIMIZE OVERLAP between ranges for')
		print('#  different rates of occurrence (the comment in each line).')
		print('#')
		print('# The overlap can be reduced by increasing the number of samples, or by increasing')
		print('#  failchance by a few orders of magnitude.')

		PER_LINE = 6
		for (prob, orders), (lo, hi) in zip(probs, ranges):
			# put orders into compact string form in case they have been tuplefied
			orders = [''.join(x) for x in orders]

			for i in range(0, len(orders), PER_LINE):
				range_args = range_fmt.format(lo, hi)
				order_args = ', '.join(repr(x) for x in orders[i:i+PER_LINE]) # repr to quote
				print('self.validate_range({}, {}) # p = {:0.4f}'.format(range_args, order_args, prob))

		print('#            END AUTOGENERATED CODE')
		print('######################################################')

	#  HERE ENDS A BUNCH OF UGLY MACHINERY
	# ======================================


	@staticmethod
	def __test_uniform__probs():
		return [(1./24, [''.join(x) for x in itertools.permutations('abcd')])]

	def test_uniform(self):
		g = nx.Graph()
		g.add_path('abcd')

		self.prepare(g, s_mode=uniform(), nsamples=7500)

		# Do codegen instead?
		if False:
			probs = self.__test_uniform__probs()
			self.codegen_range_checks(probs, failchance=1e-10)
			assert False, 'Did codegen instead, see output'

		self.collect()

		######################################################
		# BEGIN CODE AUTOGENERATED BY codegen_range_checks()
		# Parameters:
		#    Number of samples: 7500
		#    Chance of spontaneous failure: ~1e-10
		#
		# The first two numbers of each line are the "range" of accepted counts in the
		#  final distribution. Ideally you want to MINIMIZE OVERLAP between ranges for
		#  different rates of occurrence (the comment in each line).
		#
		# The overlap can be reduced by increasing the number of samples, or by increasing
		#  failchance by a few orders of magnitude.
		self.validate_range((209, 427), 'abcd', 'abdc', 'acbd', 'acdb', 'adbc', 'adcb') # p = 0.0417
		self.validate_range((209, 427), 'bacd', 'badc', 'bcad', 'bcda', 'bdac', 'bdca') # p = 0.0417
		self.validate_range((209, 427), 'cabd', 'cadb', 'cbad', 'cbda', 'cdab', 'cdba') # p = 0.0417
		self.validate_range((209, 427), 'dabc', 'dacb', 'dbac', 'dbca', 'dcab', 'dcba') # p = 0.0417
		#            END AUTOGENERATED CODE
		######################################################


	@staticmethod
	def __test_by_deleted_neighbors__probs(weights):
		assert len(weights) == 3

		def p(s, i):
			# Probability of selecting the node with index i, given a "status string" s.
			# Each character in s is either '.' (indicating that the node cannot be chosen)
			#  or an integer (representing which 'kind' that node is).
			# In this particular case, the 'kind' is equal to the number of chosen neighbors.
			counts = tuple(s.count(x) for x in '012')
			kind = int(s[i])
			return weighted_prob(weights, counts, kind)

		# return probs, which is a list of tuples (p, [orderings with probability p])
		return [
			(p('0000',0) * p('.100',1) * p('..10',2),  ['abcd','dcba']),
			(p('0000',0) * p('.100',1) * p('..10',3),  ['abdc','dcab']),
			(p('0000',0) * p('.100',2) * p('.2.1',1),  ['acbd','dbca']),
			(p('0000',0) * p('.100',2) * p('.2.1',3),  ['acdb','dbac']),
			(p('0000',0) * p('.100',3) * p('.11.',1),  ['adbc','adcb','dabc','dacb']),
			(p('0000',1) * p('1.10',0) * p('..10',2),  ['bacd','cdba']),
			(p('0000',1) * p('1.10',0) * p('..10',3),  ['badc','cdab']),
			(p('0000',1) * p('1.10',2) * p('1..1',0),  ['bcad','bcda','cbad','cbda']),
			(p('0000',1) * p('1.10',3) * p('1.2.',0),  ['bdac','cadb']),
			(p('0000',1) * p('1.10',3) * p('1.2.',2),  ['bdca','cabd']),
		]

	def test_by_deleted_neighbors(self):
		g = nx.Graph()
		g.add_path('abcd')

		weights = [1., 20., 100.]
		self.prepare(g, s_mode=by_deleted_neighbors(weights), nsamples=7500)

		# Do codegen instead?
		if False:
			probs = self.__test_by_deleted_neighbors__probs(weights)
			self.codegen_range_checks(probs, failchance=1e-10)
			assert False, 'Did codegen instead, see output'

		self.collect()

		######################################################
		# BEGIN CODE AUTOGENERATED BY codegen_range_checks()
		# Parameters:
		#    Number of samples: 7500
		#    Chance of spontaneous failure: ~1e-10
		#
		# The first two numbers of each line are the "range" of accepted counts in the
		#  final distribution. Ideally you want to MINIMIZE OVERLAP between ranges for
		#  different rates of occurrence (the comment in each line).
		#
		# The overlap can be reduced by increasing the number of samples, or by increasing
		#  failchance by a few orders of magnitude.
		self.validate_range((1400, 1853), 'abcd', 'dcba') # p = 0.2165
		self.validate_range(( 700, 1051), 'bacd', 'cdba') # p = 0.1161
		self.validate_range(( 331,  594), 'bcad', 'bcda', 'cbad', 'cbda') # p = 0.0610
		self.validate_range((  31,  143), 'abdc', 'dcab') # p = 0.0108
		self.validate_range((  25,  129), 'acbd', 'dbca') # p = 0.0095
		self.validate_range((   9,   90), 'badc', 'cdab') # p = 0.0058
		self.validate_range((   9,   89), 'adbc', 'adcb', 'dabc', 'dacb') # p = 0.0057
		self.validate_range((   6,   82), 'bdca', 'cabd') # p = 0.0051
		self.validate_range((   0,   43), 'acdb', 'dbac') # p = 0.0019
		self.validate_range((   0,   30), 'bdac', 'cadb') # p = 0.0010
		#            END AUTOGENERATED CODE
		######################################################

	def test_by_deleted_neighbors_limit(self):
		# test that `by_deleted_neighbors` continues to use the last element in weights
		#  for overly large counts (rather than... say... crashing)
		g = nx.Graph()
		g.add_path('abcd')

		# with only one weight, all should use the same weight
		weights = [20.]
		self.prepare(g, s_mode=by_deleted_neighbors(weights), nsamples=7500)
		self.collect()

		# Use codegen from uniform

		######################################################
		# BEGIN CODE AUTOGENERATED BY codegen_range_checks()
		# Parameters:
		#    Number of samples: 7500
		#    Chance of spontaneous failure: ~1e-10
		#
		# The first two numbers of each line are the "range" of accepted counts in the
		#  final distribution. Ideally you want to MINIMIZE OVERLAP between ranges for
		#  different rates of occurrence (the comment in each line).
		#
		# The overlap can be reduced by increasing the number of samples, or by increasing
		#  failchance by a few orders of magnitude.
		self.validate_range((209, 427), 'abcd', 'abdc', 'acbd', 'acdb', 'adbc', 'adcb') # p = 0.0417
		self.validate_range((209, 427), 'bacd', 'badc', 'bcad', 'bcda', 'bdac', 'bdca') # p = 0.0417
		self.validate_range((209, 427), 'cabd', 'cadb', 'cbad', 'cbda', 'cdab', 'cdba') # p = 0.0417
		self.validate_range((209, 427), 'dabc', 'dacb', 'dbac', 'dbca', 'dcab', 'dcba') # p = 0.0417
		#            END AUTOGENERATED CODE
		######################################################




