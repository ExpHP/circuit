
from __future__ import print_function


import sys
if sys.version_info[0] < 3:
	# I love how there's an official ``2to3`` tool but no ``3to2``, so that
	#   any platform agnostic distribution either has to be written in python 2
	#   or tread around very carefully with a module like `six`.
	# It's no wonder why Python 3 is almost as successful as Perl 6.
	print('This package does not support python2. Try `python3 setup.py`', file=sys.stderr)
	sys.exit(1)

import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#==========================================================================

def shell(s, echo=True):
	if echo:
		print(s)
	subprocess.Popen(s, shell=True, executable='/bin/bash')

# (a reprehensible hack; is there no way to add a "rule" to setup?)
if "cleanall" in sys.argv:
	print("Deleting cython files...")
	shell("rm -f defect/ext/*.c")
	shell("rm -f detect/ext/*.so")

	# Now do a normal clean
	sys.argv[sys.argv.index('cleanall')] = 'clean'

#==========================================================================


extensions = [
	Extension(
		language='c++',
		name='defect.ext.cXorBasis',
		undef_macros=['NDEBUG'],
		sources=[
			'defect/ext/cXorBasis.pyx', # or '.cpp' if not using cythonize
			'defect/ext/xorbasis.cpp',
		],
		include_dirs=['.'],
		extra_compile_args=[
			'-Wall',
			'--pedantic',
			'--std=c++11',
		],
	),
]
extensions = cythonize(extensions)

#note: flags prior to setup.py were CPPFLAGS="-Wall --pedantic -O2 --std=c++11 -fPIC -I/usr/include/python3.4 -DNDEBUG"

setup(
	name='Defect',
	version = '0.0',
	description = 'Defect trial runner',
	url = 'https://github.com/ExpHP/circuit',
	author = 'Michael Lamparski',
	author_email = 'lampam@rpi.edu',

	ext_modules = extensions,
	packages = ['defect'],
)
