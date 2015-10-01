
DESC = '''
General circuit generation script

Calls one of the predefined circuit generation scripts.
'''

import argparse, sys, os

from . import hex_bridge
from . import mos2
from . import square
from . import triangular

ENTRY_POINTS = {
	'hexagonal':hex_bridge.main,
	'mos2':mos2.main,
	'square':square.main,
	'triangular':triangular.main,
}

def main():
	progname, *argv = sys.argv
	progname = os.path.split(progname)[1]
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

