#!/usr/bin/env python3

# Simple piece of code which uses CycleBasisBuilder in more or less its intended fashion,
#  with a hook for progress reporting (because it can kind of take a while)

import sys

from graph.cyclebasis.builder import CycleBasisBuilder

__all__ = [
	'main',
	'build_cyclebasis',
	'build_cyclebasis_terminal',
	'STAGE_PROVIDED',
	'STAGE_FALLBACK',
]

THOROUGH_DEFAULT = False

STAGE_PROVIDED = 10
STAGE_FALLBACK = 20

def main(argv):
	print("Running from CLI yet implemented.")
	sys.exit(1)

# cycles - an arbitrarily long list of "suggested" cycles (usually desirable cycles of short length),
#          which need not all be linearly dependent. (a cycle is represented with a list of (mostly)
#          non-repeating vertices, which may optionally be terminated with a copy of the first vertex,
#          e.g. [2,4,6,8,2] and [2,4,6,8] are equivalent)
#
# fallback - a complete cyclebasis of the graph, generated through any means necessary.
#            This is used to fill any holes in the provided cycles (and also to determine the correct
#            length of the cycle basis!)
#
# thorough - If False, stop checking when len(fallback) cycles have been found.
#            If True, continue checking, and produce an error if more than len(fallback) cycles are found.
#            (this may indicate a poorly-generated fallback cyclebasis, or the presence of nonexistent
#             vertices/edges in the suggested cycles)
def build_cyclebasis(cycles, fallback, thorough=THOROUGH_DEFAULT, progress_callback=lambda d:None):
	cycles = list(cycles)
	fallback = list(fallback)

	cbbuilder = CycleBasisBuilder()

	def emit_progress(*, stage_id, stage_length, stage_current):
		progress_callback({
			'stage_id': stage_id,
			'stage_length': stage_length,
			'stage_current': stage_current,
			'total_found': len(cbbuilder.cycles),
			'total_needed': len(fallback),
		})

	def add_cycles_from(lst, stage):
		emit_progress(
			stage_id = stage,
			stage_length = len(lst),
			stage_current = 0,
		)
		for i,cycle in enumerate(lst):

			if cycle[0] != cycle[-1]:
				cycle.append(cycle[0])

			# Behold: The single line in this script which actually does anything
			cbbuilder.add_if_independent(cycle)

			# this is done after checking the cycle and before exiting the loop so that
			#  the last iteration looks like 30000/30000 (as opposed to 29999/30000)
			emit_progress(
				stage_id = stage,
				stage_length = len(lst),
				stage_current = i+1,
			)

			if (not thorough) and len(cbbuilder.cycles) >= len(fallback):
				break

			if len(cbbuilder.cycles) > len(fallback):
				# Could be due to a bad fallback, or the suggested cycles might contain invalid edges;
				# Make no assumptions.
				raise RuntimeError('Found more than len(fallback) linearly independent cycles!')

	add_cycles_from(cycles, stage=STAGE_PROVIDED)
	add_cycles_from(fallback, stage=STAGE_FALLBACK)

	assert len(cbbuilder.cycles) == len(fallback)
	return cbbuilder.cycles

# Quick "frontend" for build_cyclebasis which reports incremental progress to stdout if verbose=True.
def build_cyclebasis_terminal(cycles, fallback, thorough=THOROUGH_DEFAULT, verbose=False):
	cycles = list(cycles)
	fallback = list(fallback)
	if verbose:
		print('Generating cyclebasis of length {} from {} provided cycles'.format(len(fallback), len(cycles)))

		stage_names = {
			STAGE_PROVIDED:'provided',
			STAGE_FALLBACK:'fallback',
		}
		def callback(d):
			# numeric lengths for alignment
			len_total = len(str(d['total_needed']))
			len_stage = len(str(d['stage_length']))

			found_ratio = '{{0[total_found]:{0}d}} / {{0[total_needed]:{0}d}}'.format(len_total).format(d)
			stage_name = stage_names[d['stage_id']]
			stage_ratio = '{{0[stage_current]:{0}d}} / {{0[stage_length]:{0}d}}'.format(len_stage).format(d)

			sys.stdout.write('\r')
			sys.stdout.write('Cycles found: {} .  '.format(found_ratio))
			sys.stdout.write('Searching {} cycles (Progress: {})'.format(stage_name, stage_ratio))

			# start new line between stages
			if d['stage_current'] == d['stage_length']:
				sys.stdout.write('\n')

			sys.stdout.flush()

	else:
		def callback(d):
			pass

	return build_cyclebasis(cycles, fallback, thorough, callback)

if __name__ == '__main__':
	main(sys.argv)

