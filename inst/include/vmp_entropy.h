#ifndef ENTROPY_H
#define ENTROPY_H

#include <RcppArmadillo.h>

// Entropy in natural parameters
double GaussianEntropy(arma::vec& eta);
double InverseGammaEntropy(arma::vec& eta);
double InverseChiSquareEntropy(arma::vec& eta);
double InverseWishartEntropy(arma::vec& eta);
double BernoulliEntropy(double eta);


#endif