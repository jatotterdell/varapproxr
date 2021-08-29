// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rmath.h>


//' Calculate vec^(-1)
//' 
//' @param v A vector of dimension d^2 by 1
// [[Rcpp::export]]
arma::mat inv_vectorise(arma::vec v) {
  int d = sqrt(v.n_elem);
  arma::mat V = arma::reshape(v, d, d);
  return(V);
}


//' Calculate vech of a matrix
//'
//' @param X A square matrix of dimension d by d
// [[Rcpp::export]]
arma::vec vech(arma::mat X) {
  if(!X.is_square()) {
    throw std::range_error("X must be square");  
  }
  arma::uvec lower_indices = arma::trimatl_ind(arma::size(X));
  arma::vec lower_part = X(lower_indices);
  return(lower_part);
}


//' Calculate inverse of vech for a vector
//'
//' @param v A vector of dimension d(d+1)/2 of lower triangular elements
// [[Rcpp::export]]
arma::mat inv_vech(arma::vec v) {
  int l = arma::size(v)[0];
  double d = 0.5*(sqrt(8*l + 1) - 1);
  double intpart;
  if(modf(d, &intpart) != 0.0) {
    throw std::range_error("v must be dimension d(d+1)/2 for square matrix d x d.");  
  }
  arma::mat out(d, d, arma::fill::zeros);
  arma::uvec lower_indices = arma::trimatl_ind(size(out));
  out.elem(lower_indices) = v;
  return(arma::symmatl(out));
}


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
//' Constructs as dense block-diagonal matrix from a list (field) of matrices.
//' 
//' @param x A list of matrices
// [[Rcpp::export]]
arma::mat blockDiag(arma::field<arma::mat>& x) {
  unsigned int n = x.n_rows;
  int dimenRow = 0;
  int dimenCol = 0;
  arma::ivec dimvecRow(n);
  arma::ivec dimvecCol(n);
  for(unsigned int i=0; i<n; i++) {
    dimvecRow(i) = x(i,0).n_rows; 
    dimvecCol(i) = x(i,0).n_cols; 
    dimenRow += dimvecRow(i);
    dimenCol += dimvecCol(i);
  }
  arma::mat X(dimenRow, dimenCol, arma::fill::zeros);
  int idxRow=0;
  int idxCol=0;
  for(unsigned int i=0; i<n; i++) {
    X.submat(idxRow, idxCol, idxRow + dimvecRow(i) - 1, idxCol + dimvecCol(i) - 1 ) = x(i,0);
    idxRow += dimvecRow(i);
    idxCol += dimvecCol(i);
  }
  return(X);
}


arma::mat repeatBlockDiag(arma::mat& x, arma::mat& R) {
  int n = x.n_rows;
  int k = R.n_rows;
  int dimenRow = 0;
  int dimenCol = 0;
  arma::ivec dimvecRow(n);
  arma::ivec dimvecCol(n);
  for(int i = 0; i < n; i++) {
    dimvecRow(i) = k; 
    dimvecCol(i) = k; 
    dimenRow += dimvecRow(i);
    dimenCol += dimvecCol(i);
  }
  arma::mat X(dimenRow, dimenCol, arma::fill::zeros);
  int idxRow=0;
  int idxCol=0;
  for(unsigned int i=0; i<n; i++) {
    X.submat(idxRow, idxCol, idxRow + dimvecRow(i) - 1, idxCol + dimvecCol(i) - 1 ) = x(i,0);
    idxRow += dimvecRow(i);
    idxCol += dimvecCol(i);
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


//' Solve two level sparse matrix problem.
//' 
//' @param x A list of matrices
// [[Rcpp::export]]
Rcpp::List solve_two_level_sparse(
  arma::vec a1,
  arma::mat A11,
  arma::field<arma::vec> a2,
  arma::field<arma::mat> A22,
  arma::field<arma::mat> A12
) {
  
  int p = a1.n_rows;
  int m = a2.n_rows;
  if(p != A11.n_rows)
    Rcpp::stop("Dimension mismatch in a1 and A11");
  
  // outputs
  arma::vec x1(size(a1));
  arma::mat X11(size(A11));
  arma::field<arma::vec> x2(m);
  arma::field<arma::mat> X22(m);
  arma::field<arma::mat> X12(m);
  
  arma::field<arma::mat> invA22(m);
  arma::vec omega1 = a1;
  arma::mat Omega1 = A11;

  for(int i = 0; i < m; i++) {
    invA22(i) = inv(A22(i));
    omega1 -= A12(i)*invA22(i)*a2(i);
    Omega1 -= A12(i)*invA22(i)*A22(i).t();
  }
  X11 = inv(Omega1);
  x1 = A11*omega1;

  for(int i = 0; i < m; i++) {
    x2(i) = invA22(i) * (a2(i) - A12(i).t() * x1);
    X12(i) = -(invA22(i)*A12(i).t()*X11).t();
    X22(i) = invA22(i)*(1.0 - A12(i).t()*X12(i));
  }
  
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("x1") = x1,
    Rcpp::Named("X11") = X11,
    Rcpp::Named("x2") = x2,
    Rcpp::Named("X22") = X22,
    Rcpp::Named("X12") = X12
  );
  return out;
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
