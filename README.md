# `defect`

Models an atomic network as a circuit.
It computes currents along such networks, and can introduce "defects" into these networks
(which may be represented by a variety of modifications to the circuit). Where reasonable,
such updates are made in-place using algorithms of lower complexity than would be required
to recompute the currents entirely from scratch.

This is still *very much* in alpha and I consider very little of the codebase to be "stable."

# Dependencies

* [`networkx`](https://github.com/networkx/networkx)
* [`numpy`](https://github.com/numpy/numpy) and [`scipy`](https://github.com/scipy/scipy)
* [`pytoml`](https://github.com/avakar/pytoml)
* [`dill`](https://pypi.python.org/pypi/dill)
* [`cython`](http://cython.org/)
* a `c++` compiler with support for `c++11` features

**You need `python3`**.  There are currently no plans to support `python2`.
In the ubuntu Trusty Tahr repositories, you will find packages prefixed with `python3` (e.g. `python3-networkx`).
Or if you prefer using Pip to install python modules, try `pip3 install`.

# Installing/using

1. Clone this repo: `git clone https://github.com/ExpHP/defect/`
2. Run `python3 setup.py install`.
3. Ask me how to run stuff because there aren't really any details that I feel are stable enough to put in a document that I only update once every blue moon.

[Bugs go here](https://github.com/ExpHP/defect/issues).

# Documentation

...it's pretty low priority at the moment, considering that the total number of people to have ever touched this project is still 1.

# Testing

The testing situation has been improving as of late!  Enter the repo root and run `nosetests3`.

The tests themselves are scattered around a bit.
There are some `unittest`-style tests here, some doctests there... and in some older parts of the code there are actually unit tests for helper functions sprinkled around haphazardly at the module level (so that they run automatically when the library is loaded)!
The test coverage is incomplete, but improving.

