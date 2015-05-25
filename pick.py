
import random
from bisect import bisect_left
from collections import Counter

__all__ = [
	'uniform',
	'by_occurences',
	'weighted',
]

def uniform(it):
	return random.choice(list(it))

identity = lambda x: x

# allitems: set of all possible items, including those with zero occurrences
# weight: a map from count to weight, either via
#   - a function f(count) -> weight
#   - a type with __getitem__ (such as a list [weight] or a dict {count: weight})
# If not specified, weight will be proportional to number of occurences.
def by_occurences(it, allitems=None, weight=identity):

	# allow func, list, or dict
	if not hasattr(weight, '__call__'):
		if hasattr(weight, '__getitem__'):
			weight = weight.__getitem__
		else:
			raise TypeError('unexpected type for weight func: %s' % type(weight))

	counts = Counter(it)
	if allitems is None:
		allitems = set(counts)

	# items missing from allitems is considered a logic error (only purpose of allitems
	#  is to allow items to have zero count)
	extras = set(counts) - set(allitems)
	if len(extras) > 0:
		first = next(iter(extras))
		raise ValueError('Item in list but not in allitems: %r' % (first,))

	return weighted(allitems, (weight(counts[x]) for x in allitems))

# A weighted random selection.
# Items of weight == 0 (this includes floating point 0.0 and -0.0) are considered excluded from the list.
def weighted(it, weights, rng=random):
	items = list(it)
	weights = list(weights)

	if len(weights) != len(items):
		raise ValueError('Received lists of unequal length (items: %d, weights: %d)' % (len(items), len(weights)))
	if len(weights) == 0:
		raise ValueError('Cannot choose from empty list!')

	# sort by weight for floating point stability concerns
	weights,items = zip(*sorted(zip(weights,items), key = lambda pi: pi[0]))

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

