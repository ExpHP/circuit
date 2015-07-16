
# cXorBasis.pyx: Cython bindings for xorbasis.hpp

# This provides a C++ implementation of the XorBasisBuilder class, which manages an RREF bit
#   matrix for the purposes of building e.g. a cycle basis.
# XorBasisBuilder is currently a class with very limited offerings; it basically only offers
#   precisely the functionality required by CycleBasisBuilder, and no more than that.
# (that is to say, perhaps it might be better named "CycleBasisBuilder_Impl" :P)

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

# Q: Why are you declaring meaningful names like `column_t` and `identity_t` and then not using them?
# A: They're just here for you, the reader.  I had every intention of using them, but as it turns
#    out, cython has trouble performing automatic list->vector conversion for vectors of custom types:
#
#    cdef extern from *:
#        ctypedef column_t X '__pyx_t_9cXorBasis_column_t'
#                ^
#    ------------------------------------------------------------
#    
#    vector.from_py:37:13: 'column_t' is not a type identifier
#
#
# Q: ...um, then why do you also define and use `uint`?  Wouldn't that be similarly broken?
# A: Funny you ask, because no, `vector[uint]` works just fine!  Though I highly suspect it isn't using
#    my `uint`, but rather one that it managed to find defined somewhere else.
#    Come to think of it, gee, I sure hope it has the same definition as mine...
#
# Q: Wait, you're not even sure that the `uint` it uses is the same as your `uint`? Holy smokes,
#    Batman, that sounds dangerous!  Why not just use `unsigned int`?
# A: Oh yeah, that.  See, I also had trouble declaring a `vector[unsigned]` (something about `unsigned`
#    not being a known type) or a `vector[vector[unsigned int]]` (expected `]`, found `int`).
#
# Q: Plan to bring these bugs up with the cython team?
# A: Eh... odds are 5% these bugs are legitimate, 95% I'm just a bonehead.
ctypedef unsigned uint
ctypedef unsigned column_t
ctypedef size_t   identity_t

cdef extern from "defect/ext/xorbasis.h":
    cdef cppclass _XorBasisBuilder:
        _XorBasisBuilder()
        size_t add(vector[uint])
        vector[size_t] add_many(vector[vector[uint]])
        pair[bint,size_t] add_if_linearly_independent(vector[uint])
        vector[vector[size_t]] get_zero_sums()
#        void remove_from_each_zero_sum(vector[size_t])
        void remove_ids(vector[size_t])
#        void remove_linearly_dependent_ids()
#        bint has_linearly_dependent_rows()


cdef class XorBasisBuilder:
    cdef _XorBasisBuilder *thisptr

    def __cinit__(self):
        self.thisptr = new _XorBasisBuilder()

    def __dealloc__(self):
        del self.thisptr

    def add(self, row):
        cdef vector[uint] vec = list(row)
        return self.thisptr.add(vec)

    def add_many(self, it):
        cdef vector[vector[uint]] vec = list(list(x) for x in it)
        return self.thisptr.add_many(vec)

    def add_if_linearly_independent(self, row):
        cdef vector[uint] vec = list(row)
        cdef pair[bint,size_t] p = self.thisptr.add_if_linearly_independent(vec)
        return (bool(p.first), p.second)

    def get_zero_sums(self):
        return self.thisptr.get_zero_sums()

#    def remove_from_each_zero_sum(self, ids):
#        cdef vector[size_t] vec = list(ids)
#        self.thisptr.remove_from_each_zero_sum(vec)

#    def get_linearly_dependent_ids(self):
#        return self.thisptr.get_linearly_dependent_ids()

    def remove_ids(self, it):
        cdef vector[size_t] ids = list(it)
        self.thisptr.remove_ids(ids)

#    def remove_linearly_dependent_ids(self):
#        self.thisptr.remove_linearly_dependent_ids()
