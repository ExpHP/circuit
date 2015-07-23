
import itertools
import tempfile
import pickle

def window2(it):
	'''
	Get (overlapping) adjacent pairs from an iterable.

	>>> x = [1,2,3,10,7]
	>>> derivative = [b-a for (a,b) in window2(x)]
	>>> derivative
	[1, 1, 7, -3]
	>>> list(window2([1])) # no "pairs"
	[]
	>>> list(window2([]))  # likewise
	[]
	'''
	it = iter(it) # allow next() to consume elements

	try: prev = next(it)
	except StopIteration:  # 0-length list
		return

	for x in it:
		yield (prev,x)
		prev = x

def partition(pred, it):
	'''
	Splits an iterable into two based on a predicate.

	Returns ``(ayes, nays)``.

	>>> compare10 = lambda x: x>10
	>>> greater, lesser = partition(compare10, [1,4,77,20,5])
	>>> greater
	[77, 20]
	>>> lesser
	[1, 4, 5]
	'''
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

	>>> list(zip_matching_length([1,2,3], [4,5,6]))
	[(1, 4), (2, 5), (3, 6)]
	>>> list(zip_matching_length([1,2,3], [4,5]))
	Traceback (most recent call last):
	  ...
	ValueError: zip_matching_length called on iterables of mismatched length
	>>> # edge cases
	>>> list(zip_matching_length([],[],[]))
	[]
	>>> list(zip_matching_length())
	[]
	'''
	sentinel = object()
	zipped = list(map(tuple, itertools.zip_longest(*arrs, fillvalue=sentinel)))

	# all arrays were length 0
	if len(zipped) == 0:
		return []

	if sentinel in zipped[-1]:
		raise ValueError('zip_matching_length called on iterables of mismatched length')
	return zipped

def zip_variadic(*its):
	'''
	Variant of ``zip`` that returns tuples of varying length.

	Rather than stopping with the shortest iterable, this will produce tuples of variable
	length, with one element for each iterable that has not yet finished.

	>>> list(zip_variadic([1,2,3,4], [11,22], [111,222,333]))
	[(1, 11, 111), (2, 22, 222), (3, 333), (4,)]
	>>> list(zip_variadic())
	[]
	'''
	sentinel = object()
	def without_fill(xs):
		return filter(lambda x: x is not sentinel, xs)
	return (tuple(without_fill(xs)) for xs in itertools.zip_longest(*its, fillvalue=sentinel))


def edictget(d, e):
	'''
	Index a dict which takes undirected graph edges as keys.

	The dict is assumed to only store 1 value for each edge. (behavior is undefined on a dict
	which has different values for e.g. `(1,2)` and `(2,1)`).

	>>> d = {(1,4):'a'}
	>>> edictget(d, (1,4))
	'a'
	>>> edictget(d, (4,1))
	'a'
	'''
	return d[e[::-1] if e[::-1] in d else e]


class TempfileWrapper:
	'''
	Wrap a function to take some arguments via temporary files.

	``TempfileWrapper`` exists to aid the creation of worker processes (via
	e.g. ``multiprocessing.Pool.map`` or ``dill``) that take large objects
	as input.  Constructing one is similar to calling ``functools.partial``,
	and the wrapped function is accessible via the ``func`` field:

	>>> add = lambda a,b,c: a + b + c
	>>> wrapper = TempfileWrapper(add, 10, 20)
	>>> wrapper.func(12)
	42
	>>> wrapper = TempfileWrapper(add, c=3) # keyword arguments ok too
	>>> wrapper.func(1333, 1)
	1337

	When created, a ``TempfileWrapper`` pickles each argument to a temporary
	file.  The wrapped function reads these temporary files to obtain the
	arguments when called. To assist in cleanup, the temporary files will
	automatically be closed once no more references to the TempfileWrapper
	exist. This means you must be mindful not to accidentally "orphan" the
	wrapped function:

	>>> def bad_idea():
	...     wrapper = TempfileWrapper(lambda x: x, 3)
	...     return wrapper.func # last reference to wrapper falls out of scope
	...
	>>> func = bad_idea()
	>>> func()
	... # doctest: +IGNORE_EXCEPTION_DETAIL
	Traceback (most recent call last):
	  ...
	FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmp9rxjs5gq'

	PORTABILITY CONCERNS: Works on Linux.  May fail to work on other operating
	systems due to the fact that ``TempfileWrapper`` keeps open file handles in
	write mode on each temp file (a limitation of ``tempfile.NamedTemporaryFile``).
	'''
	def __init__(self, f, *args, **kw):
		self.__tempfiles = []

		def make_temp(obj):
			tmp = tempfile.NamedTemporaryFile('wb')
			pickle.dump(obj, tmp, pickle.HIGHEST_PROTOCOL)
			tmp.flush()

			self.__tempfiles.append(tmp)

			return tmp.name

		argpaths = [make_temp(x) for x in args]
		kwpaths  = {k:make_temp(v) for k,v in kw.items()}

		# (note how wrapped does NOT reference `self`.  We don't want this closure to
		#  close over the temp files, so keep it this way. :P)
		def wrapped(*moreargs, **morekw):
			def read_temp(path):
				with open(path, 'rb') as f:
					return pickle.load(f)

			allargs = [read_temp(x) for x in argpaths]
			allkw   = {k:read_temp(v) for k,v in kwpaths.items()}

			allargs += moreargs
			allkw.update(morekw)
			return f(*allargs, **allkw)

		self.func = wrapped

def dict_inverse(d):
	if len(set(d.values())) != len(d): raise ValueError('dictionary is not one-to-one!')
	return {v:k for k,v in d.items()}

# takes a dict `d` whose values are iterable (and of equal length N) and returns
#  dicts d1,d2,...,dN such that dn[k] == d[k][n]
def unzip_dict(d):
	''' Turn a dict with iterable values into a tuple of dicts.

	The values are required to be of equal length.

	>>> roman, caps = unzip_dict({'a':('I','A'), 'b':('II','B')})
	>>> roman == {'a':'I', 'b':'II'}
	True
	>>> caps == {'a':'A', 'b':'B'}
	True
	>>> unzip_dict({'a':['a1','a2'], 'b':['b1']}) # lengths must match
	Traceback (most recent call last):
	  ...
	ValueError: zip_matching_length called on iterables of mismatched length
	'''
	zipped = zip_matching_length(*d.values())
	return tuple({k:v for k,v in zip(d.keys(), x)} for x in zipped)

def zip_dict(*ds):
	'''
	Combine dicts with matching keys into a single dict with tuple values.

	>>> roman = {'a':'I', 'b':'II'}
	>>> caps = {'a':'A', 'b':'B'}
	>>> zip_dict(roman, caps) == {'a':('I','A'), 'b':('II','B')}
	True
	'''
	if len(ds) == 0: return {}

	keys = set(ds[0]) # used to synchronize order of lists
	values = []
	for d in ds:
		if set(d) != keys:
			raise ValueError('Keys do not match.')
		values.append([d[k] for k in keys])

	zipped = zip_matching_length(*values)
	return {k:tuple(x) for k,x in zip(keys, zipped)}

if __name__ == '__main__':
	import doctest
	doctest.testmod()

