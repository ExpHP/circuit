
import toml

class Config:
	'''
	The defect trial TOML "config file", which is pretty unremarkable at this
	point and only contains a couple of graph-related things such as which
	nodes are deletable and which nodes belong to the battery; basically,
	any property of the graph which is only needed by the trial runner (and
	therefore isn't part of the ``.circuit`` file).

	Bit of a wart in the design at this point, really.
	'''
	def __init__(self, measured_edge=None, no_defect=None):
		self.__edge = None
		self.__no_defect = None
		if measured_edge is not None: self.set_measured_edge(*measured_edge)
		if no_defect is not None: self.set_no_defect(no_defect)

	def set_measured_edge(self, s, t): self.__edge = [s,t]
	def set_no_defect(self, d):     self.__no_defect = list(d)

	def get_measured_edge(self): return tuple(self.__edge)
	def get_no_defect(self):  return list(self.__no_defect)

	@classmethod
	def from_file(cls, path):
		with open(path) as f:
			s = f.read()
		return cls.deserialize(s)

	def save(self, path):
		s = self.serialize()
		with open(path, 'w') as f:
			f.write(s)

	@classmethod
	def deserialize(cls, s):
		d = toml.loads(s)
		measured_edge = tuple(d['general']['measured_edge'])
		no_defect = list(d['general']['no_defect'])
		return cls(measured_edge, no_defect)

	def serialize(self):
		d = {
			'general': {
				'measured_edge': self.__edge,
				'no_defect': self.__no_defect,
			},
		}
		return toml.dumps(d)

