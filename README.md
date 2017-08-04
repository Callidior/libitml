libitml
=======

This is an efficient `C++` implementation of [Information-Theoretic Metric Learning][1] (Davis et al., ICML 2017),
based on the [Python implementation by CJ Carey][2].

It requires the [Eigen library][3] of version at least 3.1 and has been tested with `gcc` 4.8 and 5.3 and `Eigen` 3.3.

Using `gcc`, the library can be compiled as follows:

    g++ -march=native -Wall -O3 --shared -fPIC -o libitml.so libitml.cc -fopenmp

Make sure to have `Eigen` in your include path or add a corresponding `-I` flag.

The main `itml()` function is implemented in `C++`, but `C`-style interfaces `itml_float()` and `itml_double()` are available,
so that it can be easily used from any other programming language. A Python interface function is provided as well.

[1]: http://www.cs.utexas.edu/users/pjain/itml/
[2]: https://github.com/all-umass/metric-learn/blob/master/metric_learn/itml.py
[3]: http://eigen.tuxfamily.org/