

class BfsVisitor:
	def start_vertex(self, g, v):
		''' Invoked on the root of a bfs tree. '''
		pass

	def discover_vertex(self, g, v):
		''' Invoked on a vertex when seen as the target of a tree edge. '''
		pass

	def examine_vertex(self, g, v):
		''' Invoked on a vertex when popped from the queue. '''
		pass

	def finish_vertex(self, g, v):
		''' Invoked after all edges of a vertex have been examined. '''
		pass

	def tree_edge(self, g, e, source):
		''' Invoked when a tree edge is examined. '''
		pass

	def non_tree_edge(self, g, e, source):
		''' Invoked when a non-tree edge is examined. '''
		pass

	def examine_edge(self, g, e, source):
		''' Invoked on all out-edges of a vertex. '''
		pass

class DfsVisitor:
	def start_vertex(self, g, v):
		''' Invoked on the root of a dfs tree. '''
		pass

	def discover_vertex(self, g, v):
		''' Invoked on the target of a tree edge. '''
		pass

	def finish_vertex(self, g, v):
		''' Invoked after all edges of a vertex have been examined. '''
		pass

	def tree_edge(self, g, e, source):
		''' Invoked when a tree edge is examined. '''
		pass

	def non_tree_edge(self, g, e, source):
		''' Invoked when a back, forward, or cross-edge is examined.

		On undirected graphs, this will NOT be invoked on tree edges when they
		are examined backwards. '''
		pass

	def finish_edge(self, g, e, source):
		''' Invoked when retracing a tree edge after finishing a vertex.

		`source` is the original source vertex (not the vertex we just finished). '''
		pass

# Makes a visitor, overriding its member methods with functions provided by cb_dict.
def _make_visitor(cls, cb_dict):
	obj = cls()
	for k,v in cb_dict.items():
		if k not in cls.__dict__:
			raise KeyError('cannot override method {}; no such method'.format(k))
		obj.__dict__[k] = v
	return obj

def make_bfs_visitor(**kwargs):
	_make_visitor(BfsVisitor, kwargs)

def make_dfs_visitor(**kwargs):
	_make_visitor(DfsVisitor, kwargs)

# Handles the visitor and **callbacks arguments, either by returning the visitor,
#  or by constructing one from the callbacks.
def visitor_from_visitor_args(cls, visitor, callbacks):
	if visitor is None:
		visitor = _make_visitor(cls, callbacks)
	elif len(callbacks) > 0:
		raise RuntimeError('Received both a visitor and callbacks!')

	return visitor

def bfs_full(graph, visitor=None, **callbacks):
	visitor = visitor_from_visitor_args(BfsVisitor, visitor, callbacks)

	remaining = set(graph.vertices())
	visited   = set()

	while len(remaining) > 0:
		visited = graph.bfs_rooted(remaining.pop(), visited, visitor)
		remaining.difference_update(visited)

def dfs_full(graph, visitor=None, **callbacks):
	visitor = visitor_from_visitor_args(DfsVisitor, visitor, callbacks)

	remaining = set(graph.vertices())
	visited   = set()

	while len(remaining) > 0:
		visited = graph.dfs_rooted(remaining.pop(), visited, visitor)
		remaining.difference_update(visited)

def bfs_rooted(graph, root, visited=None, visitor=None, **callbacks):
	visitor = visitor_from_visitor_args(BfsVisitor, visitor, callbacks)

	if visited is None: visited = set()
	else:               visited = set(visited)

	_bfs_rooted_impl(graph, root, visitor, visited)
	return visited

def dfs_rooted(graph, root, visited=None, visitor=None, **callbacks):
	visitor = visitor_from_visitor_args(DfsVisitor, visitor, callbacks)

	if visited is None: visited = set()
	else:               visited = set(visited)

	_dfs_rooted_impl(graph, root, visitor, visited)
	return visited

def _dfs_rooted_impl(graph, root, visitor, visited):
	# Written in an iterative fashion due to Python's limited support for recursion.
	# Here be dragons

	visitor.start_vertex(graph, root)

	# stack contains:
	# (src_edge, vertex, out_edge_iter)
	stack = [(None, root, iter(graph.out_edges(root)))]

	while len(stack) > 0:
		(src_edge, v, out_edges) = stack[-1]

		visited.add(v)

		# Get one edge
		try: e = next(out_edges)

		# No edge found
		except StopIteration:
			visitor.finish_vertex(graph, v)

			if src_edge is not None:
				source = graph.edge_source_given_target(src_edge, v)
				visitor.finish_edge(graph, src_edge, source)

			# Return to previous vertex
			stack.pop()

		# Edge found
		else:

			# Ignore the edge that brought us here (for undirected graphs)
			if e == src_edge:
				continue

			target = graph.edge_target_given_source(e, v)

			if target in visited:
				visitor.non_tree_edge(graph, e, v)

			else:
				visitor.tree_edge(graph, e, v)
				visitor.discover_vertex(graph, target)

				# Visit target on next iteration
				stack.append((e, target, iter(graph.out_edges(target))))


	
