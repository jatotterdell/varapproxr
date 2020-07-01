#include <RcppArmadillo.h>
#include <Rmath.h>

//' Woodbury matrix identity
//'
//' (A + BCD)^{-1}
//' 
//' @param A mat A
//' @param B mat B
//' @param C mat C
//' @param D mat D
// [[Rcpp::export]]
arma::mat woodbury(
    arma::mat& A, 
    arma::mat& B, 
    arma::mat& C, 
    arma::mat& D) {
  arma::mat Ainv = inv(A);
  return(Ainv + Ainv*B*inv(inv(C) + D*Ainv*B)*D*inv(A));
}

//' Convert arma::vec to Rcpp::NumericVector
//' 
//' @param x A vector
// [[Rcpp::export]]
Rcpp::NumericVector arma2vec(arma::vec x) {
  return Rcpp::NumericVector(x.begin(), x.end());
}



//' Construct block-diagonal matrix from list of matrices
//' 
//' @param x A list of matrices
// [[Rcpp::export]]
arma::mat blockDiag(arma::field<arma::mat>& x) {
  unsigned int n = x.n_rows;
  int dimen = 0;
  arma::ivec dimvec(n);
  for(unsigned int i=0; i<n; i++) {
    dimvec(i) = x(i,0).n_rows; 
    dimen += dimvec(i);
  }
  
  arma::mat X(dimen, dimen, arma::fill::zeros);
  int idx=0;
  for(unsigned int i=0; i<n; i++) {
    X.submat(idx, idx, idx + dimvec(i) - 1, idx + dimvec(i) - 1 ) = x(i,0);
    idx = idx + dimvec(i);
  }
  
  return(X);
}

//' Right-bind columns of matrices from in list.
//' 
//' @param x A list of matrices
// [[Rcpp::export]]
arma::mat bind_cols(arma::field<arma::mat>& x) {
  unsigned int n = x.n_rows;
  int c_dimen = 0;
  int r_dimen = 0;
  arma::ivec c_dimvec(n);
  arma::ivec r_dimvec(n);
  
  for(unsigned int i=0; i<n; i++) {
    c_dimvec(i) = x(i,0).n_cols; 
    r_dimvec(i) = x(i,0).n_rows; 
    c_dimen += c_dimvec(i);
    r_dimen += r_dimvec(i);
  }  
  
  // Need to check that all matrices have same number of rows...
  // Currently no check for this...
  
  arma::mat X(r_dimvec(0), c_dimen, arma::fill::zeros);
  int idx=0;
  for(unsigned int i=0; i<n; i++) {
    X.submat(0, idx, r_dimvec(0) - 1, idx + c_dimvec(i) - 1) = x(i,0);
    idx = idx + c_dimvec(i);
  }
  
  return(X);
}


//' Multivariate Normal Entropy
//' 
//' Calculate and return the entropy for multivariate distribution
//' with supplied covariance matrix.
//' 
//' @param S Covariance matrix 
// [[Rcpp::export]]
double mvn_entropy(arma::mat& S) {
  int d = S.n_rows;
  return 0.5*(d*(1 + log(2*M_PI)) + real(log_det(S)));
}


//' Inverse Gamma Entropy
//' 
//' Calculate and return the entropy for inverse gamma distribution
//' with supplied shape and scale.
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_entropy(double a, double b) {
  return a + log(b) + lgamma(a) - (a + 1)*R::digamma(a);
}


//' Inverse Gamma E[x]
//' 
//' Calculate and return the entropy for inverse gamma distribution
//' with supplied shape and scale.
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E(double a, double b) {
  double ret = 0.0;
  if(a > 1)
    ret = b / (a - 1);
  return ret;
}


//' Inverse Gamma E[1/x]
//' 
//' Calculate and return the entropy for inverse gamma distribution
//' with supplied shape and scale.
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E_inv(double a, double b) {
  return a/b;
}

//' Inverse Gamma E[log(x)]
//' 
//' Calculate and return the entropy for inverse gamma distribution
//' with supplied shape and scale.
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E_log(double a, double b) {
  return log(b) - R::digamma(a);
}

//' Evaluate standard normal cdf for matrix of variates
//' 
//' @param m A matrix of variates 
// [[Rcpp::export]]
arma::mat pnorm_mat(arma::mat& m) {
  int p = m.n_cols;
  int n = m.n_rows;
  arma::mat out(n, p);
  
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < p; j++) {
      out(i, j) = R::pnorm(m(i, j), 0.0, 1.0, 1, 0);
    }
  }
  return out;
}

//' Evaluate standard normal density for matrix of variates
//' 
//' @param m A matrix of variates 
// [[Rcpp::export]]
arma::mat dnorm_mat(arma::mat& m) {
  int p = m.n_cols;
  int n = m.n_rows;
  arma::mat out(n, p);
  
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < p; j++) {
      out(i, j) = R::dnorm(m(i, j), 0.0, 1.0, 0);
    }
  }
  return out;
}

// Integration constants for quadrature
const arma::vec MS_p = {0.003246343272134, 0.051517477033972,
                        0.195077912673858, 0.315569823632818,
                        0.274149576158423, 0.131076880695470,
                        0.027912418727972, 0.001449567805354};
const arma::vec MS_s = {1.365340806296348, 1.059523971016916, 
                        0.830791313765644, 0.650732166639391,
                        0.508135425366489, 0.396313345166341,
                        0.308904252267995, 0.238212616409306};
