
import itertools
import tempfile
import pickle

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


class TempPickle:
	'''
	An object pickled to a temporary file.

	This exists to aid the creation of worker threads (via e.g. ``multiprocessing.Pool.map``)
	that take a large object.  Instead of passing the object directly to workers, create a
	``TempPickle`` from it and pass the filename (accessible via the ``.path`` attribute)
	to the workers. Workers should call the static method ``TempPickle.read(path)`` to
	obtain the object.

	The actual file on disk will be deleted once no references to the TempPickle exist,
	allowing for proper cleanup regardless of exceptions.  However, this makes TempPickle
	ill-suited for asynchronous communication between concurrently running processes.

	PORTABILITY CONCERNS:  Works fine on Unix, but may unusable on e.g. Windows due to
	the fact that TempPickle keeps an open file handle on the object.
	'''
	def __init__(self, obj):
		self.__temp = tempfile.NamedTemporaryFile('wb')
		pickle.dump(obj, self.__temp, pickle.HIGHEST_PROTOCOL)
		self.__temp.flush()
		# do NOT close __temp, as doing so will delete the file early

	@property
	def path(self):
		''' File path to the pickled object. '''
		return self.__temp.name

	@staticmethod
	def read(path):
		''' Obtain an object from the path. '''
		return pickle.load(path)

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
	zipped = zip_matching_length(*d.values())
	return [{k:v for k,v in zip(d.keys(), x)} for x in zipped]

_d1,_d2 = unzip_dict({'a':(1,2),'b':(3,4)})
assert _d1 == {'a': 1, 'b': 3}
assert _d2 == {'a': 2, 'b': 4}
assertRaises(ValueError, unzip_dict, {'a':(1,2),'b':(3,)})

# takes dicts d1,d2,...,dN with matching keys and returns a dict `d`
#  of tuples   d[k] == (d1[k], d2[k], ... dN[k])
def zip_dict(*ds):
	if len(ds) == 0: return {}

	keys = set(ds[0]) # used to synchronize order of lists
	values = []
	for d in ds:
		if set(d) != keys:
			raise ValueError('Keys do not match.')
		values.append([d[k] for k in keys])

	zipped = zip_matching_length(*values)
	return {k:tuple(x) for k,x in zip(keys, zipped)}

assert zip_dict({'a': 1}, {'a': 2}) == {'a': (1,2)}

if __name__ == '__main__':
	import doctest
	doctest.testmod()

