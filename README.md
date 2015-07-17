# `circuit`

Models an atomic network as a circuit.
It computes currents along such networks, and can introduce "defects" into these networks
(which may be represented by a variety of modifications to the circuit). Where reasonable,
such updates are made in-place using algorithms of lower complexity than would be required
to recompute the currents entirely from scratch.

This is still *very much* in alpha and I consider very little of the codebase to be "stable."

(Not to mention it is currently littered with random little things like plot generating scripts
which aren't really supposed to be here, but are anyways because I haven't had much success
writing a `setup.py` (more on that below) and I need those scripts to have access to the libs)

# Dependencies

* `networkx`
* `numpy`
* `scipy`

Also, the vast majority of the code targets `python3`.
There are currently no plans to support `python2`.
In the ubuntu Trusty Tahr repositories, you will find packages prefixed with `python3` (e.g. `python3-networkx`).
Or if you prefer using Pip to install python modules, try `pip3 install`.

# Installing/using

Oh, you actually want to use it?

...wait, *_really?_*

Um, well, best of luck with that.
I'm still trying to figure out how to set up `distutil`s for this in a reasonable fashion....
For now--and I know this is terrible, but--you can:

* Add the root of repo to your `PYTHONPATH`
* Run `build-xorbasis.sh` to build the C++ extension `cXorBasis`
* Ask me how to run stuff because there aren't really any details that I feel are stable enough to put in a document that I only update once every blue moon.

Or B) uh... help me with distribution?
