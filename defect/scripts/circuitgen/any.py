
DESC = '''
General circuit generation script

Calls one of the predefined circuit generation scripts.
'''

import argparse, sys

from . import hex_bridge
from . import mos2
from . import square

ENTRY_POINTS = {
	'graphene':hex_bridge.main,
	'mos2':mos2.main,
	'square':square.main,
}

def main():
	progname, *argv = sys.argv
	parser = argparse.ArgumentParser(prog=progname, description=DESC,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument('script', choices=ENTRY_POINTS, help='script to run')
	parser.add_argument('ARGS', nargs=argparse.REMAINDER, help='args to script')

	args = parser.parse_args(argv)

	# HACK: stick our argument into progname so that it is
	#  included in the usage string when the next script runs
	progname = '{} {}'.format(progname, args.script)
	# Run the requested script
	ENTRY_POINTS[args.script](progname, args.ARGS)

if __name__ == '__main__':
	main()

