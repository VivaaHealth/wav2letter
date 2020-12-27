# wav2letter++

Notable's fork of [wav2letter v0.2](https://github.com/facebookresearch/wav2letter/tree/v0.2)

This fork is modified to build Python bindings to support simple inference like the [`SimpleStreamingASRExample`](inference/inference/examples/SimpleStreamingASRExample.cpp) example. It could probably be simplified a lot to accomplish only this without dragging in so much else from wav2letter, but someone will have to figure out how. 

To build, from a clone of this repo, run:
```docker-compose up --build --detach```
This will place into the `dist` folder a Python wheel (something like `wav2letter-0.0.2-cp36-cp36m-linux_x86_64.whl`). (Before rebuilding, be sure to `docker-compose down`.)

The wheel targets Ubuntu 18.04. To install it, run `pip3 install wav2letter-𝑒𝑡𝑐.whl` (you may first need to do `pip3 install wheel`). 

The inference library also requires some [depencencies](https://github.com/facebookresearch/wav2letter/wiki/Dependencies#3-additional-dependencies) to be installed; `apt-get install -y libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev` should suffice.

See [`test_inference.py`](inference/wav2letter/test_inference.py) for an example of how to use from Python. 

The Python bindings and associated C++ code are in [`_inference.cpp`](inference/wav2letter/_inference.cpp), so start there if you need to modify anything. 

To make a new release:
* Update the version number in [`setup.py`](inference/setup.py).
* Build
* Tag the release in GitHub and attach the wheel file. 