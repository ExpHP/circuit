
'''
A JSON-based graph format for ``networkx`` graphs.

Supports directed and undirected graphs (though not multigraphs),
allows nodes to be integer or (unicode) string, and includes all
types of attributes supported by networkx (graph, node, and edge).

The format is designed with homogenous data structures in mind.
In contrast to networkx's attribute model, all nodes are expected
to have the same set of attributes, and likewise with edges.
This requirement allows for a decent reduction in size; attributes
are stored as lists, where the ``i``th element belongs to the ``i``th
node or edge.
'''

import networkx as nx
import json

HIGHEST_VERSION = 1

# TODO tests

def write_networkx(g, path):
	'''
	Save a networkx ``Graph`` or ``DiGraph`` using parallel list format.
	'''
	if g.is_multigraph():
		raise ValueError('Multigraphs not supported.') # too much hassle to do them right

	_validate_graph_attributes(g)

	nodes = list(g.nodes())
	edges = list(g.edges())
	gattrs = dict(g.graph)

	nattrs = dict()
	if len(nodes) > 0:
		node_set = set(nodes)
		for attr in g.node[nodes[0]]:
			d = nx.get_node_attributes(g, attr)
			assert set(d) == node_set
			nattrs[attr] = [d[n] for n in nodes]

	eattrs = dict()
	if len(edges) > 0:
		edge_set = set(edges)
		s0, t0 = edges[0]
		for attr in g.edge[s0][t0]:
			d = nx.get_edge_attributes(g, attr)
			assert set(d) == edge_set
			eattrs[attr] = [d[e] for e in edges]

	d = {
		'formatver':  HIGHEST_VERSION,
		'directed':   g.is_directed(),
		'nodes':      nodes,
		'edges':      edges,
		'graph_attr': gattrs,
		'edge_attr':  eattrs,
		'node_attr':  nattrs,
	}
	with open(path, 'w') as f:
		json.dump(d, f)


def read_networkx(path):
	'''
	Read a networkx ``Graph`` or ``DiGraph`` saved in parallel list format.
	'''
	with open(path) as f:
		d = json.load(f)

	if d['formatver'] > HIGHEST_VERSION:
		raise RuntimeError('Unsupported file format version {}'.format(d['formatver']))

	if d['directed']: g = nx.DiGraph()
	else:             g = nx.Graph()

	nodes = list(d['nodes'])
	edges = list(map(tuple, d['edges']))

	if len(nodes) != len(set(nodes)): raise RuntimeError('Duplicate node!')
	# TODO The same error for edges (have to account for undirected/directed and blehhhck its boring)

	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	for attr, values in d['node_attr'].items():
		if len(values) != len(nodes):
			raise RuntimeError('Node attribute {} has incorrect number of elements!'.format(repr(attr)))
		nx.set_node_attributes(g, attr, {n:value for n,value in zip(nodes, values)})

	for attr, values in d['edge_attr'].items():
		if len(values) != len(edges):
			raise RuntimeError('Edge attribute {} has incorrect number of elements!'.format(repr(attr)))
		nx.set_edge_attributes(g, attr, {e:value for e,value in zip(edges, values)})

	for attr, value in d['graph_attr'].items():
		g.graph[attr] = value

	assert _validate_graph_attributes(g) # post-condition
	return g


# Verifies that any node/edge attributes in g are completely defined for all nodes/edges.
# Raises an error or returns True (for use in assertions).
def _validate_graph_attributes(g):
	all_node_attributes = set()
	for v in g:
		all_node_attributes.update(g.node[v])

	for attr in all_node_attributes:
		if len(nx.get_node_attributes(g, attr)) != g.number_of_nodes():
			raise ValueError('node attribute {} is set on some nodes but not others'.format(repr(attr)))

	all_edge_attributes = set()
	for s,t in g.edges():
		all_edge_attributes.update(g.edge[s][t])

	for attr in all_edge_attributes:
		if len(nx.get_edge_attributes(g, attr)) != g.number_of_edges():
			raise ValueError('edge attribute {} is set on some edges but not others'.format(repr(attr)))

	return True

