
import random
from bisect import bisect_left

__all__ = [
	'uniform',
	'by_deleted_neighbors',
]

class uniform:
	def selection_func(self, choices, g, initial_g):
		return random.choice(list(choices))

	def info(self):
		return {'mode': 'uniform'}

class by_deleted_neighbors:
	def __init__(self, weights):
		# NOTE: the caller is expected to provide an array that is at least as long
		#       as the maximum degree of a deletable vertex in the graph.
		self.weights = weights

	def selection_func(self, choices, g, initial_g):
		choices = list(choices) # make reusable iter

		max_nbrs = initial_g.degree()
		cur_nbrs = g.degree()
		choice_weights = (self.weights[max_nbrs[v] - cur_nbrs[v]] for v in choices)

		return pick_weighted(choices, choice_weights)

	def info(self):
		return {
			'mode': 'by deleted neighbor count',
			'weights': self.weights,
		}

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

