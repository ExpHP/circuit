#!/usr/bin/env python3

import os
import sys
import math
import random
import time
import functools
try:
	import cProfile as profile
except ImportError:
	import profile

from defect.trial_runner import TrialRunner

import networkx as nx
import numpy as np
import json
import toml

from multiprocessing import Pool
from defect.util import multiprocessing_dill, TempfileWrapper

from defect.circuit import MeshCurrentSolver, CircuitBuilder, load_circuit
from defect import graph
import defect.graph.path as vpath

from defect.components import node_selection, node_deletion, cyclebasis_provider

from defect import util

# TODO: maybe implement subparsers for these, and put in the node_selection/deletion
#   modules since these need to be updated for each new mode
SELECTION_MODES = {
	'uniform':  node_selection.uniform(),
	'bigholes': node_selection.by_deleted_neighbors([1,10**3,10**4,10**7]),
}
# XXX temporary hack - lambdas to handle options for deletion modes because
# XXX  I don't want to deal with subparsers yet. Not all options apply
# XXX  to all modes
DELETION_MODES = { # XXX
	'remove':   lambda **kw: node_deletion.annihilation(kw['radius']),
	'multiply': lambda **kw: node_deletion.multiply_resistance(kw['strength'], False, kw['radius']),
	'assign':   lambda **kw: node_deletion.multiply_resistance(kw['strength'], True,  kw['radius']),
}

def main():
	import argparse
	parser = argparse.ArgumentParser()

	# some "type" callbacks for simple validations.
	# Do not use this to test conditions based on factors external to the program
	#  (such as path being writable), because these won't get called on default values!
	nonnegative_int = validating_conversion(int, lambda x: x>=0, 'is not a nonnegative integer')
	positive_int = validating_conversion(int, lambda x: x>=1, 'is not a positive integer')

	parser.add_argument('input', type=str, help='.gpickle file of networkx graph')
	parser.add_argument('--verbose', '-v', action='store_true')
	parser.add_argument('--jobs', '-j', type=positive_int, default=1, help='number of trials to run in parallel')
	parser.add_argument('--trials', '-t', type=positive_int, default=1, help='number of trials to do total')
	parser.add_argument('--steps', '-s', type=nonnegative_int, default=None, help='number of steps per trial')
	parser.add_argument('--config', '-c', type=str, default=None, help='Path to defect trial config TOML. '
		'Default is derived from circuit (BASENAME.defect.toml)')
	parser.add_argument('--substeps', '-x', type=positive_int, default=1, help='number of defects added per step')
	parser.add_argument('--output-json', '-o', type=str, default=None, help='output file. Default is derived '
		'from circuit (BASENAME.results.json).')
	parser.add_argument('--output-pstats', '-P', type=str, default=None, help='Record profiling info (implies --jobs 1)')
	parser.add_argument('--selection-mode', '-S', type=str, default='uniform', choices=SELECTION_MODES, help='TODO')
	parser.add_argument('--deletion-mode', '-D', type=str, required=True, choices=DELETION_MODES, help='TODO')
	parser.add_argument('--alltheway', dest='end_on_disconnect', action='store_false', help='always have a trial continue until there are no nodes left, even if the circuit is disconnected')

	# cyclebasis options
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--cyclebasis-cycles', type=str, default=None, help='Path to cyclebasis file. '
		'Default is derived from circuit (BASENAME.cycles)')
	group.add_argument('--cyclebasis-planar', type=str, default=None, help='Path to planar embedding info, which '
		'can be provided in place of a .cycles file for planar graphs.  Default is BASENAME.planar.gpos.')

	# XXX temporary hack - options for configuring deletion modes because
	# XXX  I don't want to deal with subparsers yet. Not all options apply
	# XXX  to all modes
	parser.add_argument('--Dstrength', type=float, default=10.)
	parser.add_argument('--Dradius', type=int, default=1)

	args = parser.parse_args(sys.argv[1:])

	if (args.output_pstats is not None) and args.jobs != 1:
		die('--output-pstats/-P is limited to --jobs 1\n'
			'In other words: No multiprocess profiling!')

	# common behavior for filepaths which are optionally specified
	basename = drop_extension(args.input)
	def get_optional_path(userpath, extension, argname):
		autopath = basename + extension
		if userpath is not None:
			return userpath
		if args.verbose:
			notice('Note: %s not specified!  Trying %r', argname, autopath)
		return autopath

	args.config = get_optional_path(args.config, '.defect.toml', '--config')
	args.output_json = get_optional_path(args.output_json, '.results.json', '--output-json')

	# save the user some grief; fail early if output paths are not writable
	for path in (args.output_json, args.output_pstats):
		if path is not None:
			die_if_not_writable(path)

	# load input files
	config = Config.from_file(args.config)

	g = load_circuit(args.input)
	cbprovider = cbprovider_from_args(basename, args)

	# setup
	runner = TrialRunner()
	runner.set_initial_circuit(g)
	runner.set_initial_choices(set(g) - set(config.get_no_defect()))
	runner.set_initial_cycles(cbprovider.new_cyclebasis(g))
	runner.set_measured_edge(*config.get_measured_edge())
	runner.set_selection_mode(SELECTION_MODES[args.selection_mode])
	runner.set_deletion_mode(DELETION_MODES[args.deletion_mode](strength=args.Dstrength, radius=args.Dradius)) # XXX
	if args.steps is not None:
		runner.set_step_limit(args.steps)
	else:
		runner.unset_step_limit()
	runner.set_defects_per_step(args.substeps)
	runner.set_end_on_disconnect(args.end_on_disconnect)

	# The function that worker threads will invoke
	cmd_once = functools.partial(TrialRunner.run_trial, verbose=args.verbose)

	# Bind to the runner instance via a temp file as it may be extremely large.
	# (this limits the number of simultaneous in-memory copies to the number of
	#  RUNNING jobs, rather than one for each trial that WILL run)
	cmd_wrapper = TempfileWrapper(cmd_once, runner) # MUST keep a living reference to the wrapper!!
	cmd_once = cmd_wrapper.func

	# Callbacks for reporting when a trial starts/ends
	def onstart(trial, ntrials):
		if args.verbose:
			notice('Starting trial %s (of %s)', trial+1, ntrials)
	def onend(trial, ntrials):
		pass

	if args.jobs == 1:
		cmd_all = lambda: run_sequential(cmd_once, times=args.trials, onstart=onstart, onend=onend)
	else:
		cmd_all = lambda: run_parallel(cmd_once, threads=args.jobs, times=args.trials, onstart=onstart, onend=onend)

	if args.output_pstats is not None:
		assert args.jobs == 1
		cmd_all = wrap_with_profiling(args.output_pstats, cmd_all)

	info = {}

	info['selection_mode'] = runner.selection_mode.info()
	info['defect_mode'] = runner.deletion_mode.info()
	info['cyclebasis_gen'] = cbprovider.info()

	info['process_count'] = args.jobs
	info['profiling_enabled'] = (args.output_pstats is not None)

	info['time_started'] = int(time.time())
	info['trials'] = cmd_all() # do eeeet
	info['time_finished'] = int(time.time())

	assert isinstance(info['trials'], list)

	if args.output_json is not None:
		s = json.dumps(info)
		with open(args.output_json, 'w') as f:
			f.write(s)

def cbprovider_from_args(basename, args):
	# The order to check is
	# User Cycles --> User Planar --> Auto Cycles --> Auto Planar --> "Nothing found"
	def from_cycles(path): return cyclebasis_provider.from_file(path)
	def from_planar(path): return cyclebasis_provider.planar.from_gpos(path)

	for userpath, constructor in [
		(args.cyclebasis_cycles, from_cycles),
		(args.cyclebasis_planar, from_planar),
	]:
		if userpath is not None:
			die_if_not_readable(userpath)
			return constructor(userpath)

	if args.verbose:
		notice('Note: "--cyclebasis-cycles" or "--cyclebasis-planar" not specified. Trying defaults...')
	for autopath, constructor in [
		(basename + '.cycles',      from_cycles),
		(basename + '.planar.gpos', from_planar),
	]:
		if os.path.exists(autopath):
			if args.verbose:
				notice('->Found possible cyclebasis info at %r', autopath)
			die_if_not_readable(autopath)
			return constructor(autopath)
	die('Cannot find cyclebasis info. You need a .cycles or .planar.gpos file.\n'
		'For more info search for "--cyclebasis" in the program help (-h).')
	sys.exit(1)

def die_if_not_readable(path):
	try:
		with open(path, 'r') as f:
			pass
	except IOError as e:
		die("Could not verify %r as readable:\n%s", path, e)

# NOTE unintentional side-effect: creates an empty file if nothing exists
def die_if_not_writable(path):
	try:
		with open(path, 'a') as f:
			pass
	except IOError as e:
		die("Could not verify %r as writable:\n%s", path, e)

def run_sequential(f,*,times,onstart=None,onend=None):
	result = []
	for i in range(times):
		if onstart: onstart(i, times)  # for e.g. reporting
		result.append(f())
		if onend: onend(i, times)
	return result

def run_parallel(f,*,threads,times,onstart=None,onend=None):

	# Give each trial a unique seed
	baseseed = time.time()
	arglists = [(i, baseseed+i) for i in range(times)]

	def run_with_seed(args):
		i,seed = args

		random.seed(seed)

		if onstart: onstart(i, times)  # for e.g. reporting
		result = f()
		if onend: onend(i, times)

		return result

	p = Pool(threads)
	return multiprocessing_dill.map(p, run_with_seed, arglists, chunksize=1)

def wrap_with_profiling(pstatsfile, f):
	def wrapped(*args, **kwargs):
		p = profile.Profile()
		p.enable()
		result = f(*args, **kwargs)
		p.disable()

		try:
			p.dump_stats(pstatsfile)
		except IOError as e: # not worth losing our return value over
			warn('could not write pstats. (%s)', e)

		return result
	return wrapped

def drop_extension(path):
	head,tail = os.path.split(path)
	if '.' in tail:
		tail, _ = tail.rsplit('.', 1)
	return os.path.join(head, tail)

class Config:
	def __init__(self, measured_edge=None, no_defect=None):
		self.__edge = None
		self.__no_defect = None
		if measured_edge is not None: self.set_measured_edge(*measured_edge)
		if no_defect is not None: self.set_no_defect(no_defect)

	def set_measured_edge(self, s, t): self.__edge = [s,t]
	def set_no_defect(self, d):     self.__no_defect = list(d)

	def get_measured_edge(self): return tuple(self.__edge)
	def get_no_defect(self):  return list(self.__no_defect)

	@classmethod
	def from_file(cls, path):
		with open(path) as f:
			s = f.read()
		return cls.deserialize(s)

	def save(self, path):
		s = self.serialize()
		with open(path, 'w') as f:
			f.write(s)

	@classmethod
	def deserialize(cls, s):
		d = toml.loads(s)
		measured_edge = tuple(d['general']['measured_edge'])
		no_defect = list(d['general']['no_defect'])
		return cls(measured_edge, no_defect)

	def serialize(self):
		d = {
			'general': {
				'measured_edge': self.__edge,
				'no_defect': self.__no_defect,
			},
		}
		return toml.dumps(d)

def validating_conversion(basetype, pred, failmsg):
	import argparse
	def func(s):
		error = argparse.ArgumentTypeError(repr(s) + failmsg)
		# this is only intended for simple validation on simple types;
		# in such cases, little value is lost by substituting all exceptions
		#  with one that just names the requirements
		try: x = basetype(s)
		except Exception: raise error
		if not pred(x): raise error
		return x
	return func

# think logger.info, except the name `info` already belonged to
#  a local variable in some places
def notice(msg, *args):
	print(msg % args)

def warn(msg, *args):
	print('Warning: ' + (msg % args), file=sys.stderr)

def die(msg, *args, code=1):
	print('Fatal: ' + (msg % args), file=sys.stderr)
	sys.exit(code)

if __name__ == '__main__':
	main()
