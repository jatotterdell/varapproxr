#ifndef VMP_FRAGMENTS_H
#define VMP_FRAGMENTS_H

#include <RcppArmadillo.h>

arma::vec G_VMP(arma::vec v, arma::mat Q, arma::vec r, double s);

arma::vec GaussianPriorFragment(arma::vec mu, arma::mat Sigma);
arma::vec InverseGammaPriorFragment(double a, double b);
arma::vec InverseWishartPriorFragment(double xi, arma::mat Lambda);
arma::field<arma::vec> InverseGWishartPriorFragment(arma::mat G, arma::vec xi, arma::mat Lambda);
arma::field<arma::vec> IteratedInverseGWishartFragment(arma::mat G, arma::vec xi, arma::vec eta1_in, arma::vec eta2_in);

arma::field<arma::vec> GaussianLikelihoodFragment(arma::vec n, arma::mat XtX, arma::vec Xty, double yty, arma::vec eta1_in, arma::vec eta2_in);

arma::field<arma::vec> ExpectationGaussianSufficientStatistics(arma::vec eta);

arma::field<arma::mat> GaussianCommonParameters(arma::vec& eta);
arma::field<arma::mat> InverseGWishartCommonParameters(arma::vec& eta);

#endif