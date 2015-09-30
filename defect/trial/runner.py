
from defect.trial.node_deletion import *
from defect.trial.node_selection import *
from defect.circuit import load_circuit, MeshCurrentSolver

import defect.graph.cyclebasis

import time

class TrialRunner:
	'''
	Implements the defect trial algorithm.

	Set any parameters you want (via the provided setters), then
	invoke the method ``run_trial`` to run a single trial with those
	parameters from start to finish.

	Running a trial does not mutate the ``TrialRunner``, meaning that
	you can reuse the runner for multiple trials simply by calling
	``run_trial`` multiple times.
	'''

	# Named special values
	# These can't be ``object()``s because their identity must be
	#  preserved after pickling
	STEPS_UNLIMITED = None
	CHOICES_ALL = None
	NOT_SET = None

	def __init__(self):
		# Defaults
		self.__selection_mode = uniform()
		self.__deletion_mode  = self.NOT_SET
		# here we store the class so we can generate fresh instances (cbupdater is stateful)
		self.__cbupdater_cls  = defect.graph.cyclebasis.builder_cbupdater

		self.set_initial_choices(self.CHOICES_ALL)
		self.__initial_circuit = self.NOT_SET
		self.__initial_cycles  = self.NOT_SET
		self.__measured_edge   = self.NOT_SET

		self.set_end_on_disconnect(True)
		self.set_defects_per_step(1)
		self.set_step_limit(self.STEPS_UNLIMITED)

	#-----------------------------------------------------
	# Setters. Many of them trivial, but regardless, you are expected to use them,
	#  because unlike a member field, it's impossible to mispell a method.
	#
	# (which is more of a problem here than anywhere else in the codebase, as this
	#  object provides a lot of default values for things)

	def set_measured_edge(self, s, t):
		self.__measured_edge = (s, t)

	# FIXME take string modes and kw args
	def set_selection_mode(self, obj):
		self.__selection_mode = obj
	def set_deletion_mode(self, obj):
		self.__deletion_mode = obj

	def set_end_on_disconnect(self, val):
		assert isinstance(val, bool)
		self.__end_on_disconnect = val

	def set_defects_per_step(self, val):
		assert isinstance(val, int)
		assert val > 0
		self.__substeps = val

	def set_initial_cycles(self, cycles):
		self.__initial_cycles = list(map(list, cycles)) # deep-listify
	def set_initial_circuit(self, circuit):
		self.__initial_circuit = fastcopy(circuit)
	def set_initial_choices(self, choices):
		if choices == self.CHOICES_ALL:
			self.__initial_choices = choices
		else:
			self.__initial_choices = set(choices)
	def set_measured_edge(self, s, t):
		self.__measured_edge = (s, t)

	def unset_step_limit(self):
		self.set_step_limit(self.STEPS_UNLIMITED)
	def set_step_limit(self, val):
		assert (val is self.STEPS_UNLIMITED
			or (isinstance(val, int) and val >= 0))  # zero OK; just computes initial state
		self.__steps = val

	#-----------------------------------------------------
	# Running the trial

	def _validate_ready(self):
		for (var, name) in [
			(self.__initial_circuit, 'initial circuit'),
			(self.__initial_cycles, 'initial cycles'),
			(self.__measured_edge, 'measured edge'),
			(self.__deletion_mode, 'deletion mode'),
			(self.__selection_mode, 'selection mode'),
			(self.__cbupdater_cls, 'cyclebasis updater class'),
		]:
			if var == self.NOT_SET:
				raise RuntimeError('{} is not set'.format(name))

		for v in self.__measured_edge:
			if v not in self.__initial_circuit:
				raise RuntimeError('Measured edge contains node {} not in graph!'.format(repr(v)))

	# This method does NOT mutate any members of TrialRunner.
	# It runs a full trial from scratch.
	def run_trial(self, verbose=False):
		self._validate_ready()

		# Generate stateful objects used in trial

		# Initial graph is given directly to some object's constructors
		#  (the expectation being that they'll make a copy if they plan to modify it)
		g = self.__initial_circuit

		if self.__initial_choices is self.CHOICES_ALL:
			choices = set(self.__initial_circuit)
		else:
			choices = set(self.__initial_choices)

		# battery vertices can never have defects
		for v in self.__measured_edge:
			choices.remove(v)

		result = {}
		result['graph'] = {
			'num_deletable': len(choices),
			'num_vertices': g.number_of_nodes(),
			'num_edges': g.number_of_edges(),
		}
		result['steps'] = self._run_trial_steps(
			verbose=verbose,
			solver=MeshCurrentSolver(g, self.__initial_cycles, self.__cbupdater_cls()),
			deleter=self.__deletion_mode.deleter(g),
			selector=self.__selection_mode.selector(g),
			choice_set=choices,
		)
		return result

	# This method does NOT mutate any members of TrialRunner.
	# Any mutable arguments passed to this method are consumed; do not reuse them.
	def _run_trial_steps(self, verbose=False, *, solver, deleter, selector, choice_set):

		# output
		step_info = {'runtime':[], 'current':[], 'deleted':[]}

		max_defects = len(choice_set)

		def trial_should_end():
			return (len(choice_set) == 0 # no defects possible
				or selector.is_done() # e.g. a replay ended
				or (self.__end_on_disconnect and current == 0.))

		if self.__steps is self.STEPS_UNLIMITED:
			stepiter = unlimited_range()
		else:
			# Do steps+1 iterations because the first iteration is not a full step.
			# This way, steps=0 just does initial state, steps=1 adds one defect step, etc...
			stepiter = range(self.__steps + 1)

		for step in stepiter:
			t = time.time()

			# introduce defects
			defects = []
			if step > 0:  # first step is initial state

				if trial_should_end():
					break  # we're done, period (the final step has already been recorded)

				# Each substep introduces a defect.
				for _ in range(self.__substeps):
					if trial_should_end():
						break  # stop adding defects (but do record this final step)

					vcenter = selector.select_one(choice_set)
					choice_set.remove(vcenter)
					defects.append(vcenter)

					deleter.delete_one(solver, vcenter, cannot_touch=self.__measured_edge)

			# the big heavy calculation!
			current = solver.get_current(*self.__measured_edge)

			runtime = time.time() - t

			step_info['runtime'].append(runtime)
			step_info['current'].append(current)
			step_info['deleted'].append(defects)

			if verbose:
				notice('step: %s   time: %s   current: %s', step, runtime, current)

		return step_info

def unlimited_range(start=0, step=1):
	i = start
	while True:
		yield i
		i += step

# for extremely large objects this is more performant over copy.deepcopy.
# however it does not preserve object identity, and fails if the structure contains
#   certain "fun" things like a closure or a reference to an inner class
def fastcopy(obj):
	import pickle
	return pickle.loads(pickle.dumps(obj))

# FIXME: carryover from when the trial runner was fully contained in one module
# think logger.info, except the name `info` already belonged to
#  a local variable in some places
def notice(msg, *args):
	print(msg % args)

def warn(msg, *args):
	print('Warning: ' + (msg % args), file=sys.stderr)

def die(msg, *args, code=1):
	print('Fatal: ' + (msg % args), file=sys.stderr)
	sys.exit(code)

