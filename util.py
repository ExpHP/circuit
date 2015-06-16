
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

def edictget(d, e):
	return d[e[::-1] if e[::-1] in d else e]
