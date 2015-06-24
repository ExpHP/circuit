import dill

# produces argument to run_encoded
def encode_func(fun, args, kwargs):
	return dill.dumps((fun, args, kwargs))

def run_encoded(what):
	fun, args, kwargs = dill.loads(what)
	return fun(*args, **kwargs)

def map(pool, fun, it):
	return pool.map(run_encoded, (encode_func(fun, [x], {}) for x in it))

#def apply_async(pool, fun, args=[], kwargs={}):
#	return pool.apply_async(run_encoded, (encode_func(fun, args, kwargs),))
