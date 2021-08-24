// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rmath.h>


//' Log multivariate Gamma function
//' 
//' Log transformation of the multivariate gamma function.
//' 
//' @param p The dimension
//' @param x The value to evaluate
// [[Rcpp::export]]
double lmvgamma(double x, int p = 1) {
  if(x <= 0 || p < 1)
    Rcpp::stop("Input error");
  double lgam = 0.0;
  for(int i = 1; i <= p; i++) {
    lgam += lgamma(x + 0.5*(1 - i));
  }
  return 0.25*p*(p - 1) * log(M_PI) + lgam;
}


//' multivariate Gamma function
//' 
//' The multivariate gamma function.
//' 
//' @param p The dimension
//' @param x The value to evaluate
// [[Rcpp::export]]
double mvgamma(double x, int p = 1) {
  return exp(lmvgamma(x, p));
}


//' Multivariate digamma function
//' 
//' Derivative of the multivariate Gamma function.
//' 
//' @param p The dimension
//' @param x The value to evaluate
// [[Rcpp::export]]
double mvdigamma(double x, int p = 1) {
  if(x <= 0 || p < 1)
    Rcpp::stop("Input error");
  double out = 0.0;
  for(int i = 1; i <= p; i++) {
    out += R::digamma(x + 0.5 * (1 - i));
  }
  return out;
}


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
