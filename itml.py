""" Python wrapper for libitml """

import numpy as np
import os.path
from ctypes import *

BASEDIR = os.path.dirname(__file__)

LIB = []  # library will be loaded on first use
LIBITML_PATHS = [os.path.join(BASEDIR, 'libitml.so'),
                 os.path.join(BASEDIR, 'libitml.dll'),
                'libitml.so', 'libitml.dll', 'libitml', 'itml']

itml_error_msgs = {
    -1 : 'Given prior metric is not positive-semidefinite.',
    -2 : 'No non-trivial constraints given.',
    -3 : 'Invalid constraints given.'
}

class constraint_t(Structure):
    _fields_ = [('i', c_int), ('j', c_int)]

constraint_p = POINTER(constraint_t)


def itml(X, A0, pos_pairs, neg_pairs, th_pos, th_neg,
         return_metric = False, gamma = 1.0, max_iter = 1000, conv_th = 0.001,
         verbose = False, copy = True):
    """ Wrapper function for libitml performing Information-Theoretic Metric Learning (ITML).
    
    X: n-by-d matrix `X` containing one sample per row.
    
    A0: d-by-d matrix `A` specifying the prior metric serving as a regularizer (usually the
        identity matrix or inverse covariance). If set to None, the identity matrix will be used.
    
    pos_pairs: List of (i,j) tuples giving the indices of similar samples in `X`.
    
    neg_pairs: List of (i,j) tuples giving the indices of dissimilar samples in `X`.
    
    th_pos: Threshold for distances of similar samples. ITML enforces the given pairs of similar
            samples to have a distance less than this threshold.
    
    th_neg: Threshold for distances of dissimilar samples. ITML enforces the given pairs of dissimilar
            samples to have a distance greater than this threshold.
    
    return_metric: The algorithm actually learns the Cholesky decomposition `U` of the metric `A` with
                   `A = U^T * U`, which can be used to transform the data into a space where the Euclidean
                   distance corresponds to the learned metric.
                   If return_metric is False, this Cholesky decomposition `U` will be returned, otherwise
                   the full metric `A`.
    
    gamma: Controls the trade-off between satisyfing the given constraints and minimizing the divergence
           from the prior metric. Higher `gamma` puts more weight on the constraints, while lower `gamma`
           enforces stronger regularization.
    
    max_iter: Maximum number of iterations.
    
    conv_th_ Convergence threshold.
    
    verbose: If set to `true`, information about convergence will be written to `stderr` during learning.
    
    copy: If set to True, a copy of A0 will be made. Otherwise, that matrix may be modified in-place.
    
    Returns: The learned metric or its Cholesky decomposition, depending on the value of return_metric.
    
    Raises: In the case of invalid input arguments, ValueError is raised. OSError may be raised if
            libitml cannot be found.
    """
    
    lib = _init_lib()
    
    # Check X
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('X must be a 2-dimensional array.')
    if not X.flags['C_CONTIGUOUS']:
        X = np.ascontiguousarray(X)
    if X.dtype in (np.float32, np.float64):
        dtype = X.dtype
    else:
        dtype = np.float64
        X = X.astype(dtype)
    
    # Check A0
    if A0 is None:
        A0 = np.eye(X.shape[1], dtype=dtype)
    else:
        A0 = np.asarray(A0)
    if A0.ndim != 2:
        raise ValueError('A0 must be a 2-dimensional array.')
    elif A0.shape[0] != A0.shape[1]:
        raise ValueError('A0 must be a square matrix.')
    elif A0.shape[0] != X.shape[1]:
        raise ValueError('A0 must have the same number of dimensions as X.')
    if A0.dtype != dtype:
        A0 = A0.astype(dtype)
    elif copy:
        A0 = A0.copy()
    
    # Convert constraints to C structs
    constraints_pos = (constraint_t * len(pos_pairs))()
    for k, (i, j) in enumerate(pos_pairs):
        constraints_pos[k].i = i
        constraints_pos[k].j = j
    
    constraints_neg = (constraint_t * len(neg_pairs))()
    for k, (i, j) in enumerate(neg_pairs):
        constraints_neg[k].i = i
        constraints_neg[k].j = j
    
    # Call appropriate C function
    if dtype == np.float32:
        itml_func = lib.itml_float
        pX = X.ctypes.data_as(POINTER(c_float))
        pA = A0.ctypes.data_as(POINTER(c_float))
    else:
        itml_func = lib.itml_double
        pX = X.ctypes.data_as(POINTER(c_double))
        pA = A0.ctypes.data_as(POINTER(c_double))
    result = itml_func(
        X.shape[0], X.shape[1], pX, pA,
        len(constraints_pos), constraints_pos, len(constraints_neg), constraints_neg,
        th_pos, th_neg, return_metric=return_metric, gamma=gamma, max_iter=max_iter,
        conv_th=conv_th, verbose=verbose
    )
    
    # Handle errors
    if result < 0:
        raise ValueError(itml_error_msgs[result] if result in itml_error_msgs else 'unknown error')
    
    return A0


def _init_lib():
    
    if len(LIB) == 0:
        
        for i, path in enumerate(LIBITML_PATHS):
            try:
                lib = CDLL(path)
            except OSError:
                lib = None
                if i == len(LIBITML_PATHS) - 1:
                    raise
            if lib is not None:
                break
        
        lib.itml_float = _create_itml_cfunc(lib, 'itml_float', c_float)
        lib.itml_double = _create_itml_cfunc(lib, 'itml_double', c_double)
        LIB.append(lib)
        
    return LIB[0]


def _create_itml_cfunc(lib, name, float_type):
    
    pointer_type = POINTER(float_type)
    prototype = CFUNCTYPE(c_int, c_int, c_int, pointer_type, pointer_type, c_int, constraint_p, c_int, constraint_p,
                          float_type, float_type, c_bool, float_type, c_int, float_type, c_bool)
    return prototype(
        (name, lib),
        ((1,'n'), (1,'d'), (1,'pX'), (1,'pA'), (1,'nb_pos'), (1,'pos'), (1,'nb_neg'), (1,'neg'), (1,'th_pos'), (1,'th_neg'),
         (1,'return_metric',False), (1,'gamma',1.0), (1,'max_iter',1000), (1,'conv_th',0.001), (1,'verbose',False))
    )
