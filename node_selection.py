
import random
from bisect import bisect_left

__all__ = [
	'uniform',
	'by_deleted_neighbors',
]

class uniform:
	def selection_func(self, choices, initial_g, past_selections):
		return random.choice(list(choices))

	def info(self):
		return {'mode': 'uniform'}

class by_deleted_neighbors:
	def __init__(self, weights):
		self.weights = weights

	def selection_func(self, choices, initial_g, past_selections):
		choices = list(choices) # make reusable iter

		# count previously selected neighbors of each choice
		weight_ids = {v:0 for v in choices}
		for s in past_selections:
			for t in initial_g.neighbors(s):
				if t in weight_ids:
					weight_ids[t] += 1

		# use last element of self.weights for elements with too many deleted neighbors
		weight_ids = {v:min(weight_ids[v], len(self.weights)-1) for v in weight_ids}

		choice_weights = (self.weights[weight_ids[v]] for v in choices)

		return pick_weighted(choices, choice_weights)

	def info(self):
		return {
			'mode': 'by neighbor defect count',
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

