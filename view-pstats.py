#!/usr/bin/env python3

import sys
import io
import pstats

LIMIT_COUNT    = 0
LIMIT_FRACTION = 1

LIMIT_TOP = 0
LIMIT_BOTTOM = 1

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=str, help='pstats file from profiler output')
	parser.add_argument('--sort', type=str, help="sort key. Multiple keys can be provided separated by commas (',')). For"
		"available options see `pstats.Stats#show_stats` in the pydocs for the `profile` module")
	parser.add_argument('-n', type=int, help='limit to top N results')
	parser.add_argument('--reverse-sort', '-r', action='store_true', help='reverses sort order. This DOES affect which lines are selected '
		'by -n N, which may be undesirable.')
	parser.add_argument('--reverse-display', '-R', action='store_true', help='reverses output order, WITHOUT changing the lines selected by -n N. '
		'MAY BE UNRELIABLE!!')
	parser.add_argument('--pattern', '-p', type=str, help='regular expression to filter names by')

	args = parser.parse_args(sys.argv[1:])

	# redirect output to a StringIO to assist in reversing it later
	buf = io.StringIO()
	ps = pstats.Stats(args.input, stream=buf)

	if args.sort is not None:
		ps.sort_stats(*args.sort.split(','))

	restrictions = []
	if args.pattern is not None:
		restrictions.append(args.pattern)
	if args.n is not None:
		restrictions.append(args.n)

	if args.reverse_sort:
		ps.reverse_order()

	ps.print_stats(*restrictions)
	lines = buf.getvalue().split('\n')

	if args.reverse_display:
		# Reverse only the rows of the output table
		# FIXME this is frighteningly fragile
		start = first_index(lambda s: s.strip().startswith('ncalls'), lines) + 1
		stop  = first_index(lambda s: s.strip() == '', lines, start)
		lines[start:stop] = reversed(lines[start:stop])

	for line in lines:
		print(line)

def first_index(pred, xs, start=0):
	i = start
	while not pred(xs[i]):
		i += 1
	return i

if __name__ == '__main__':
	main()
