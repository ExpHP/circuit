
import itertools

# A scrolling 2-element window on an iterator
def window2(it):
	it = iter(it)
	prev = next(it)
	for x in it:
		yield (prev,x)
		prev = x

def partition(pred, it):
	yes,no = [],[]
	for x in it:
		if pred(x): yes.append(x)
		else:       no.append(x)
	return yes,no

# struggling to understand why unittest doesn't offer its useful
#  assertions as free functions
def assertRaises(cls, f, *args, **kw):
	try:
		f(*args,**kw)
	except cls:
		return
	assert False

def zip_matching_length(*arrs):
	'''
	Variant of ``zip`` which raises an error on iterators of inconsistent length.

	Raises ``ValueError`` if one of the input iterators produces a different number
	of elements from the rest.
	'''
	sentinel = object()
	zipped = list(map(tuple, itertools.zip_longest(*arrs, fillvalue=sentinel)))
	if sentinel in zipped[-1]:
		raise ValueError('zip_matching_length called on iterables of mismatched length')
	return zipped

def zip_variadic(*its):
	'''
	Variant of ``zip`` that returns tuples of varying length.

	Rather than stopping with the shortest iterable, this will produce tuples of variable
	length, with one element for each iterable that has not yet finished.
	'''
	sentinel = object()
	def without_fill(xs):
		return filter(lambda x: x is not sentinel, xs)
	return (tuple(without_fill(xs)) for xs in itertools.zip_longest(*its, fillvalue=sentinel))


def edictget(d, e):
	return d[e[::-1] if e[::-1] in d else e]
