
from . import traversal

class AlgorithmMixin(traversal.TraversalMixin):
	def spanning_forest(self):
		''' Returns an arbitrary spanning tree forest as a dict of {v: edgeFromParent}

		The root of each tree is omitted from the dict. '''
		backedges = {}

		def handle_tree_edge(g,e,source):
			target = g.edge_target_given_source(e,source)
			backedges[target] = e

		self.bfs_full(tree_edge = handle_tree_edge)
		return backedges

	def cycle_basis(self):
		''' Generate and return a cycle basis for the graph as a list of paths.

		A cycle basis is a minimal-size set of cycles such that any cycle in the
		graph can be defined as the XOR sum of cycles in the basis.

		Note that, generally speaking, the cycle basis of a graph is NOT unique.'''

		# Algorithm:
		# - Make a spanning tree/forest. ANY tree will do.
		# - For each non-tree edge, traverse the tree between the endpoints to complete a cycle.

		eBackEdges = self.spanning_forest()

		# helper method - returns list of edges from root to v
		def path_from_root(v):
			result = []

			while v in eBackEdges:
				result.append(eBackEdges[v])
				v = self.edge_source_given_target(eBackEdges[v], v)

			result.reverse()
			return result

		# build a cycle from each non-tree edge
		cycles = []
		eTreeEdges = set(eBackEdges.values())
		for e in self.edges():
			if e not in eTreeEdges:

				# find paths from root to each node
				source, target = self.edge_endpoints(e)
				sourcePath = path_from_root(source)
				targetPath = path_from_root(target)

				# length of path to common ancestor
				iFrom = len(common_prefix(sourcePath, targetPath))

				cycle = [e]                                # source to target (via non-tree edge)
				cycle.extend(reversed(targetPath[iFrom:])) # target to ancestor (along tree)
				cycle.extend(sourcePath[iFrom:])           # ancestor to source (along tree)

				cycles.append(cycle)

		return cycles

def common_prefix(*iterables):
	if len(iterables) == 0:
		raise ValueError('no iterables provided!')

	result = []
	for values in zip(*iterables):
		if any(v != values[0] for v in values[1:]):
			break
		result.append(values[0])

	return result

assert(common_prefix([0,'a',3,2,6,6,7], [0,'a',3,2,9,6]) == [0,'a',3,2])
assert(common_prefix([0,'a',3,2,6,6,7]) == [0,'a',3,2,6,6,7])
