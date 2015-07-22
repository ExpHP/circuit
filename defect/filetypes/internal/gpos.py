
'''
``gpos`` - Graph Position file format

Associates each node in a graph with an (x,y) or (x,y,z) position.
Used as a supplemental file to graphs.
'''

import json
from defect.util import zip_matching_length

# This exists partly because position data is *extremely* ubiquitous,
# and often shows up for many different purposes (e.g. actual spatial
# data; or positions for display purposes; or a planar embedding; etc).
#
# Sometimes these different purposes are interchangable and one set of
# xy data suffices for all of them, leading to the use of vague attribute
# names like 'pos', or 'x' and 'y' -- but this is not always the case.
#
# For what it's worth, you do NOT want to see what my other solution to
# this problem would have been. Or rather, perhaps, *I* do not want you
# to see it. :P

HIGHEST_VERSION = 1

# TODO test.

def write_gpos(pos, path):
	'''
	Write a gpos file.

	Arguments:
	  pos:  A ``dict`` mapping (integer or string) graph nodes to points
	        (as iterables of 2 or 3 elements).
	  path: Output path.
	'''
	d = {
		'formatver': HIGHEST_VERSION,
		'labels':    list(pos.keys()),
		'positions': [list(map(float, p)) for p in pos.values()],
	}
	assert len(d['labels']) == len(d['positions'])
	_validate_lengths(d['positions'])
	with open(path, 'w') as f:
		json.dump(d, f)


def read_gpos(path):
	'''
	Read a gpos file.

	Arguments:
	  path: Input path.
	Returns:
	  pos:  A ``dict`` mapping (integer or string) graph nodes to points
	        (as tuples of 2 or 3 elements).
	'''
	with open(path) as f:
		d = json.load(f)
	if d['formatver'] > HIGHEST_VERSION:
		raise RuntimeError('Unsupported file format version {}'.format(d['formatver']))

	pos = {}
	for label, position in zip_matching_length(d['labels'], d['positions']):
		pos[label] = tuple(map(float, position))

	_validate_lengths(pos.values())
	return pos


def _validate_lengths(it):
	lengths = list(map(len, it))
	if len(lengths) == 0:
		return True
	expected = lengths.pop()
	if any(x != expected for x in lengths):
		raise RuntimeError('Not all positions are of equal length!')
