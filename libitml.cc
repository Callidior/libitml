/// MIT License
/// 
/// Copyright (c) 2017 Bjoern Barz
/// 
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.

#include <math.h>
#include <limits>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>     // required for LLT::rankUpdate()


#define ITML_ERR_A0 -1
#define ITML_ERR_NO_CONSTRAINTS -2
#define ITML_ERR_INVALID_CONSTRAINTS -3


/**
* Stores the indices of two similar or dissimilar samples.
*/
typedef struct {
    int i; /**< Index of first sample. */
    int j; /**< Index of second sample. */
} itml_pair;


/**
* Learns a Mahalanobis distance metric `(x-y)^T * A * (x-y)` from given data and constraints using
* Information Theoretic Metric Learning (ITML).
*
* ITML minimizes the differential relative entropy between two multivariate Gaussians under constraints
* on the distance function, which can be formulated into a Bregman optimization problem by minimizing the
* LogDet divergence subject to linear constraints.
* Unlike some other methods, ITML does not rely on an eigenvalue computation or semi-definite programming.
*
* The constraints enforced by ITML have the following form:
* 
* - `(x-y)^T * A * (x-y) < th_pos` for two similar samples `x` and `y`
* - `(x-y)^T * A * (x-y) > th_neg` for two dissimilar samples `x` and `y`
* 
* In theory, individual thresholds could be specified for all pairs, but this implementation only supports
* constant `th_pos` and `th_neg` at the moment.
*
* Reference:  
* Jason V. Davis, Brian Kulis, Prateek Jain, Suvrit Sra, Inderjit S. Dhillon.  
* "Information-Theoretic Metric Learning."
* International Conference on Machine Learning (ITML), 2007.
* 
* @param[in] n The number of samples.
* 
* @param[in] d The number of dimensions of the data.
* 
* @param[in] pX Pointer to an n-by-d matrix `X` containing one sample per row, stored in row-major order.
* 
* @param[in,out] pA Pointer to a row-major d-by-d matrix `A` which initially contains the prior metric serving
* as a regularizer (usually the identity matrix or inverse covariance). The algorithm will update this matrix
* in-place, so that it will finally contain the learned metric or its Cholesky decomposition, depending on the
* value of `return_metric`.
* 
* @param[in] nb_pos Number of similarity constraints.
* 
* @param[in] pos Pointer to an array of `nb_pos` similarity constraints, given as pairs of indices of similar
* samples in `X`.
* 
* @param[in] nb_neg Number of dissimilarity constraints.
* 
* @param[in] neg Pointer to an array of `nb_neg` dissimilarity constraints, given as pairs of indices of
* dissimilar samples in `X`.
* 
* @param[in] th_pos Threshold for distances of similar samples. ITML enforces the given pairs of similar samples
* to have a distance less than this threshold.
* 
* @param[in] th_neg Threshold for distances of dissimilar samples. ITML enforces the given pairs of dissimilar
* samples to have a distance greater than this threshold.
* 
* @param[in] return_metric The algorithm actually learns the Cholesky decomposition `U` of the metric `A` with
* `A = U^T * U`, which can be used to transform the data into a space where the Euclidean distance corresponds to
* the learned metric. This matrix `U` will be stored in the matrix pointed to by `pA`. If, however, the actual
* metric `A` is desired, this parameter can be set to `true` to obtain `A` in the matrix pointed to by `pA`.
* 
* @param[in] gamma Controls the trade-off between satisyfing the given constraints and minimizing the divergence
* from the prior metric.
* Higher `gamma` puts more weight on the constraints, while lower `gamma` enforces stronger regularization.
* 
* @param[in] max_iter Maximum number of iterations.
* 
* @param[in] conv_th Convergence threshold.
* 
* @param[in] verbose If set to `true`, information about convergence will be written to `stderr` during learning.
* 
* @return On success, returns the number of iterations needed until convergence.
* If this is equal to `max_iter`, the algorithm terminated prematurely without reaching convergence.
* In the case of error, one of the following error codes is returned:
*   - `ITML_ERR_A0`: The given prior metric is not positive-semidefinite.
*   - `ITML_ERR_NO_CONSTRAINTS`: No non-trivial constraints have been given.
*   - `ITML_ERR_INVALID_CONSTRAINTS`: Some of the given indices of similar or dissimilar pairs are out of bounds.
*/
template<typename F>
int itml(int n, int d, const F * pX, F * pA,
         int nb_pos, const itml_pair * pos, int nb_neg, const itml_pair * neg, F th_pos, F th_neg,
         bool return_metric = false, F gamma = 1.0, int max_iter = 1000, F conv_th = 0.001, bool verbose = false)
{
    // General local variables
    int i;
    int num_pos, num_neg, num_constraints; // effective number of positive/negative constraints
    const itml_pair * pair;
    const F eps = std::numeric_limits<F>::epsilon();
    
    // Wrapper around array pointers
    typedef Eigen::Matrix<F, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    Eigen::Map<const Matrix> X(pX, n, d);
    Eigen::Map<Matrix> A(pA, d, d);
    
    // Cholesky decomposition of initial metric A0
    Eigen::LLT<Matrix, Eigen::Upper> llt(A);
    if (llt.info() != Eigen::Success)
        return ITML_ERR_A0;
    
    // Slice rows from X according to constraints
    Matrix vv(nb_pos + nb_neg, d);
    num_pos = num_neg = num_constraints = 0;
    for (i = 0, pair = pos; i < nb_pos; ++i, ++pair)
    {
        if (pair->i < 0 || pair->i >= n || pair->j < 0 || pair->j >= n)
            return ITML_ERR_INVALID_CONSTRAINTS;
        vv.row(num_constraints) = X.row(pair->i) - X.row(pair->j);
        if (vv.row(num_constraints).squaredNorm() > eps)
        {
            ++num_pos;
            ++num_constraints;
        }
    }
    for (i = 0, pair = neg; i < nb_neg; ++i, ++pair)
    {
        if (pair->i < 0 || pair->i >= n || pair->j < 0 || pair->j >= n)
            return ITML_ERR_INVALID_CONSTRAINTS;
        vv.row(num_constraints) = X.row(pair->i) - X.row(pair->j);
        if (vv.row(num_constraints).squaredNorm() > eps)
        {
            ++num_neg;
            ++num_constraints;
        }
    }
    if (num_constraints == 0)
        return ITML_ERR_NO_CONSTRAINTS;
    
    // Initialize ITML-specific variables
    int sign;
    F dist, alpha, beta, normsum, conv;
    F gamma_proj = std::isinf(gamma) ? 1 : gamma/(gamma+1);
    Vector Lv(d), Av(d);
    Vector lambda = Vector::Zero(num_constraints);
    Vector lambda_old = Vector::Zero(num_constraints);
    Vector bhat(num_constraints);
    bhat.head(num_pos).setConstant(th_pos);
    bhat.tail(num_neg).setConstant(th_neg);
    
    // Iterative optimization algorithm
    int it;
    for (it = 0; it < max_iter; ++it)
    {
        // Perform update for all constraints
        for (i = 0; i < num_constraints; ++i)
        {
            sign = (i < num_pos) ? 1 : -1;
            Lv.noalias() = llt.matrixU() * vv.row(i).transpose();
            dist = Lv.squaredNorm();
            alpha = std::min(lambda(i), sign * gamma_proj * (1/dist - 1/bhat(i)));
            lambda(i) -= alpha;
            beta = sign * alpha / (1 - sign * alpha * dist);
            bhat(i) = 1 / ((1 / bhat(i)) + sign * (alpha / gamma));
            Av.noalias() = llt.matrixU().transpose() * Lv;
            llt.rankUpdate(Av, beta);
        }
        
        // Check for convergence
        normsum = lambda.norm() + lambda_old.norm();
        if (normsum < eps)
        {
            conv = std::numeric_limits<F>::infinity();
            break;
        }
        conv = (lambda_old - lambda).cwiseAbs().sum();
        conv /= normsum;
        if (conv < conv_th)
            break;
        lambda_old = lambda;
        if (verbose)
            std::cerr << "itml iter: " << it << ", conv = " << conv << std::endl;
    }
    
    if (verbose)
    {
        if (it < max_iter)
            std::cerr << "itml converged at iter: " << it << ", conv = " << conv << std::endl;
        else
            std::cerr << "itml did not converge after " << it << " iterations, conv = " << conv << std::endl;
    }
    
    // Store computed metric or its Cholesky decomposition in pA
    A = llt.matrixU().toDenseMatrix();
    if (return_metric)
        A = A.transpose() * llt.matrixU();
    
    return it;
}


extern "C"
{

int itml_float(int n, int d, float * pX, float * pA,
               int nb_pos, const itml_pair * pos, int nb_neg, const itml_pair * neg, float th_pos, float th_neg,
               bool return_metric = false, float gamma = 1.0, int max_iter = 1000, float conv_th = 0.001, bool verbose = false)
{
    return itml(n, d, pX, pA, nb_pos, pos, nb_neg, neg, th_pos, th_neg, return_metric, gamma, max_iter, conv_th, verbose);
}

int itml_double(int n, int d, double * pX, double * pA,
                int nb_pos, const itml_pair * pos, int nb_neg, const itml_pair * neg, double th_pos, double th_neg,
                bool return_metric = false, double gamma = 1.0, int max_iter = 1000, double conv_th = 0.001, bool verbose = false)
{
    return itml(n, d, pX, pA, nb_pos, pos, nb_neg, neg, th_pos, th_neg, return_metric, gamma, max_iter, conv_th, verbose);
}

}