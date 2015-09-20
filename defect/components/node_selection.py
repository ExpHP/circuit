
import random
from abc import ABCMeta, abstractmethod
from bisect import bisect_left

__all__ = [
	'SelectionMode',
	'Selector',
	'uniform',
	'by_deleted_neighbors',
	'fixed_order',
]

# NOTE: Unlike the other trial runner components, SelectionMode currently does not
#  supply a ``from_info`` method or equality testing; this is because, at the time this
#  comment is being written, there is at least one SelectionMode (the fixed order mode)
#  which contains a potentially large data structure which is deliberately left out of
#  the ``info()`` output.
#
# These circumstances are not ideal, and it would be nice if ALL trial runner components
#  were small value-based structures. ``fixed_order`` would have to be modified to
#  reference a filepath instead.

class SelectionMode(metaclass=ABCMeta):
	'''
	A component of the defect trial runner.  Selects where defects are introduced.

	``SelectionMode`` are stateless by design, and merely communicate represent the
	selected mode.  A ``Selector`` does the actual selecting.
	'''
	@abstractmethod
	def selector(self, g):
		''' Construct a new Selector for use in a trial. '''
		pass
	@abstractmethod
	def info(self):
		''' Get representation for ``results.json``.

		Return value is a ``dict`` with at minimum a ``'mode'`` value (a string). '''
		pass

class Selector(metaclass=ABCMeta):
	'''
	Does the dirty work of SelectionMode for a trial.

	This exists to encapsulate internal state. Unlike SelectionMode, Selectors are
	allowed to be stateful (and thus only live for the duration of one trial)
	'''
	@abstractmethod
	def is_done(self):
		''' Boolean. Allows a Selector to force a trial to end early. '''
		pass
	@abstractmethod
	def select_one(self, choices):
		''' Select and return a vertex from ``choices``. '''
		pass

#--------------------------------------------------------

class uniform(SelectionMode, Selector):
	''' Makes uniformly random choices. '''
	def selector(self, g):
		return self  # stateless

	def info(self):
		return {'mode': 'uniform'}

	def is_done(self):
		return False

	def select_one(self, choices):
		return random.choice(list(choices))

#--------------------------------------------------------

class by_deleted_neighbors(SelectionMode):
	''' Weights choices by number of previously selected neighbors. '''
	def __init__(self, weights):
		self.weights = weights

	def selector(self, g):
		return _by_deleted_neighbors_Selector(self, g)

	def info(self):
		return {
			'mode': 'by neighbor defect count',
			'weights': self.weights,
		}

class _by_deleted_neighbors_Selector(Selector):
	def __init__(self, owner, g):
		self.weights = list(owner.weights)
		self.initial_g = g.copy()
		self.past_selections = []

	def is_done(self):
		return False

	def select_one(self, choices):
		choices = list(choices) # make reusable iter

		# count previously selected neighbors of each choice
		# (initial_g is used as, depending on the defect mode, those neighbors
		#  may no longer exist in g!)
		weight_ids = {v:0 for v in choices}
		for s in self.past_selections:
			for t in self.initial_g.neighbors(s):
				if t in weight_ids:
					weight_ids[t] += 1

		# use last element of self.weights for elements with too many deleted neighbors
		weight_ids = {v:min(weight_ids[v], len(self.weights)-1) for v in weight_ids}

		choice_weights = (self.weights[weight_ids[v]] for v in choices)

		v = pick_weighted(choices, choice_weights)
		self.past_selections.append(v)
		return v

#--------------------------------------------------------

class fixed_order(SelectionMode):
	''' Produce a fixed sequence of choices. (basically, replay mode) '''
	def __init__(self, order):
		self.order = list(order)

	def selector(self, g):
		return _fixed_order_Selector(self)

	def info(self):
		return {
			'mode': 'fixed order',
			# no need to share the order; it will be in the trial data
		}

class _fixed_order_Selector(Selector):
	def __init__(self, owner):
		self.order = list(owner.order)
		self.index = 0

	def is_done(self):
		return self.index >= len(self.order)

	def select_one(self, choices):
		v = self.order[self.index]
		if v not in choices:
			raise RuntimeError('Fixed selection order contains {} at index {}, '
				'which is not among the valid choices!'.format(repr(v), self.index))
		self.index += 1
		return v

#--------------------------------------------------------

# A weighted random selection.
# Items of weight == 0 (this includes floating point 0.0 and -0.0) are considered excluded from the list.
def pick_weighted(it, weights, rng=random):
	items = list(it)
	weights = list(weights)

	if len(weights) != len(items):
		raise ValueError('Received lists of unequal length (items: %d, weights: %d)' % (len(items), len(weights)))
	if len(weights) == 0:
		raise ValueError('Cannot choose from empty list!')

	# sort by weight for floating point stability concerns
	weights,items = zip(*sorted(zip(weights,items), key = lambda x: x[0]))

	cum_weights = prefix_sums(weights)

	if weights[0] < 0:
		raise ValueError('Received negative weight %f for item %r' % (weights[0], items[0]))
	if cum_weights[-1] == 0:  # no fuzzy logic here; even weights ~1e-70 can be appropriately scaled; zero cannot!
		raise ValueError('Total weight is 0 (no items to choose from!)')

	r = rng.random() * cum_weights[-1]
	i = bisect_left(cum_weights, r)
	assert weights[i] != 0
	return items[i]

def prefix_sums(xs):
	sums = []
	s = 0.0
	for x in xs:
		s += x
		sums.append(s)
	return sums

