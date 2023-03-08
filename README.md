Information-Theoretic Metric Learning (ITML) library
====================================================

This is an efficient `C++` implementation of [Information-Theoretic Metric Learning][1] (Davis et al., ICML 2007),
based on the [Python implementation by CJ Carey][2].

Installation
------------

Building this package requires the [Eigen library][3] of version at least 3.1 and has been tested with `gcc` 4.8 and 5.3 and `Eigen` 3.3.

If you installed `Eigen` to a non-standard location, you need to point the environment variable `CPLUS_INCLUDE_PATH` to it.

Compiling the library and installing the Python package can then be done as follows:

    python setup.py install

If only the C++ library is of interest, it can also be compiled manually using `gcc`:

    g++ -march=native -Wall -O3 --shared -fPIC -o libitml.so libitml.cc -fopenmp

Make sure to have `Eigen` in your include path (see above) or add a corresponding `-I` flag.

The main `itml()` function is implemented in `C++`, but `C`-style interfaces `itml_float()` and `itml_double()` are available,
so that it can be easily used from any other programming language. A Python interface function is provided as well.

[1]: http://www.cs.utexas.edu/users/pjain/itml/
[2]: https://github.com/all-umass/metric-learn/blob/master/metric_learn/itml.py
[3]: http://eigen.tuxfamily.org/
