
from ..basegraph import UndirectedGraph
import unittest


# NOTE: I still want to do something that will allow edges in paths (and possibly
#  edges returned by out_edges and friends as well) to be aware of which direction
#  they're being viewed from.
# Depending on how I accomplish this, the semantics of edge __eq__ may change,
#  (or even be entirely removed in favor of less ambiguous functions), and any test
#  that checks equality of edges will need to be revised.

class UndirectedTester(unittest.TestCase):
	def setUp(self):
		self.g = UndirectedGraph()

	# adds vertices from iterable to a graph and connects them in a circle
	def add_circle_component(self, iterable):
		lst = list(iterable)
		self.g.add_vertices(lst)
		for i in range(-1, len(lst)-1): # -1 to include last<->first edge
			self.g.add_edge(lst[i], lst[i+1])

	def test_double_vertex_onecall(self):
		with self.assertRaises(Exception) as cm:
			self.g.add_vertices(['a', 'b', 'c', 'b']) # duplicate vertex 'b'

	def test_double_vertex_twocall(self):
		self.g.add_vertices(['a', 'b'])
		with self.assertRaises(Exception) as cm:
			self.g.add_vertices(['c', 'b', 'e']) # duplicate vertex 'b'

	# Multigraph functionality
	def test_double_edge(self):
		self.g.add_vertices(['a', 'b'])
		self.g.add_edge('a','b')
		self.g.add_edge('a','b')

		# NOTE: See note about edge equality.
		es = list(self.g.edges())
		self.assertEqual(len(es), 2)
		self.assertEqual(es[0], es[0])
		self.assertEqual(es[1], es[1])
		self.assertNotEqual(es[0], es[1])

	# Undirectedness
	def test_back_edge(self):
		self.g.add_vertices(['a', 'b'])
		self.g.add_edge('a', 'b')

		# NOTE: See note about edge equality.
		a_edges = set(self.g.out_edges('a'))
		b_edges = set(self.g.out_edges('b'))
		self.assertSetEqual(a_edges, b_edges)

	def test_has_vertex(self):
		vs = [c for c in 'abcde']
		self.g.add_vertices(c for c in 'abde')

		actual   = {v:self.g.has_vertex(v) for v in vs}
		expected = {'a': True, 'b': True, 'c': False, 'd': True, 'e': True}
		self.assertDictEqual(actual, expected)

		self.g.delete_vertices(['b'])

		actual   = {v:self.g.has_vertex(v) for v in vs}
		expected = {'a': True, 'b': False, 'c': False, 'd': True, 'e': True}
		self.assertDictEqual(actual, expected)

	# checks that deleting a vertex does not orphan edges
	def test_edges_deleted_with_vertex(self):
		self.add_circle_component(c for c in 'abcdefg')

		self.assertEqual(self.g.num_edges(), 7)

		self.g.delete_vertices(['b'])

		self.assertEqual(self.g.num_edges(), 5)

	# Targets a potential issue specific to the igraph underpinnings,
	#  as igraph treats integer names as indices during lookup
	def test_integer_vertices(self):
		self.g.add_vertices(range(10))
		self.g.add_edge(9,3)
		self.g.add_edge(9,2)

		self.g.delete_vertices([2,5])

		# Test has_vertex
		self.assertFalse(self.g.has_vertex(2))
		self.assertTrue(self.g.has_vertex(9))

		# Test that 9-3 edge is still there
		es = list(self.g.out_edges(9))
		self.assertEqual(len(es),1)
		self.assertEqual(self.g.edge_target_given_source(es[0], 9), 3)

