
import unittest
import networkx as nx
import math
import random
from defect.graph.cyclebasis._planar import *
import defect.graph.path as vpath

#----------------------------------------------------

class NastyEdgeCases(unittest.TestCase):
	# As it turns out, the problems of determining a planar cycle basis and
	#  of identifying faces in a planar graph are NOT equivalent.
	# a cycle inside another, sharing a vertex

	# Two cycles sharing a single vertex, one inside the other.
	# This is an example of a biconnected graph.
	def test_triangles(self):
		g = nx.Graph()
		g.add_path([0,1,2,0,3,4,0])
		xs,ys = {}, {}
		xs[0] =  0.0; ys[0] = 0.0  # shared vertex
		xs[1] = -0.5; ys[1] = 1.0  # inner triangle
		xs[2] = +0.5; ys[2] = 1.0
		xs[3] = -1.5; ys[3] = 2.0  # outer triangle
		xs[4] = +1.5; ys[4] = 2.0

		check_known_cyclebasis(g, xs, ys, [[0,1,2,0],[0,3,4,0]])

	# a cycle inside another, connected by a 1-edge filament
	def test_hanging_diamond(self):
		g = nx.Graph()
		g.add_path([0,1,2,3,0,4,5,6,7,4])
		xs,ys = {v:0.0 for v in g}, {v:0.0 for v in g}

		xs[1] = +1.0; xs[5] = +0.5 # inner diamond
		xs[3] = -1.0; xs[7] = -0.5
		ys[0] = +1.0; ys[4] = +0.5
		ys[2] = -1.0; ys[6] = -0.5

		check_known_cyclebasis(g, xs, ys, [[0,1,2,3,0],[4,5,6,7,4]])

# The planar method produces a unique cycle basis (up to the ordering
#  of its elements), so we can test it directly against the "correct
#  result"
def check_known_cyclebasis(g, xs, ys, expected):
	def check_at_angle(angle):
		rotx, roty = rotate_coord_maps(xs, ys, angle)
		cb = planar_cycle_basis_nx(g, rotx, roty)
		assert vpath.cyclebases_equal(cb, expected)

	# Check using various rotated versions of the circuit to force the algorithm
	#  to try a variety of starting points.
	for i in range(4): check_at_angle(2.*math.pi * i * .25)
	for _ in range(6): check_at_angle(2.*math.pi * random.random())

def rotate_coord_maps(xs, ys, angle):
	sin,cos = math.sin(angle), math.cos(angle)
	newx = {v: cos*xs[v] - sin*ys[v] for v in xs}
	newy = {v: sin*xs[v] + cos*ys[v] for v in xs}
	return newx,newy

