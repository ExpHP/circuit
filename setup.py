
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
from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from Cython.Build import cythonize

extensions = [
	Extension(
		language='c++',
#		name='defect.ext.cXorBasis',  # <--- wanted this, but I'm having trouble getting it to work
		name='_defect',  # <--- yuck
		undef_macros=['NDEBUG'],
		sources=[
			'_defect.pyx',
#			'defect/ext/cXorBasis.pyx',
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

setup(
	name='Defect',
	version = '0.0',
	description = 'Defect trial runner',
	url = 'https://github.com/ExpHP/circuit',
	author = 'Michael Lamparski',
	author_email = 'lampam@rpi.edu',

	entry_points={
		'console_scripts':[
			'defect-trial = defect.trial.main:main',
			'defect-gen = defect.scripts.circuitgen.any:main',
			'defect-view = defect.scripts.plotting.circuit:main',
		],
	},

	install_requires=[
		'networkx',
		'numpy',
		'scipy',
		'pytoml',
		'dill',
	],

	ext_modules = extensions,
	packages=find_packages(), # include sub-packages
)
