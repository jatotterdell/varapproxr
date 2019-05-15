// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// mvn_entropy
double mvn_entropy(arma::mat& S);
RcppExport SEXP _varapproxr_mvn_entropy(SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type S(SSEXP);
    rcpp_result_gen = Rcpp::wrap(mvn_entropy(S));
    return rcpp_result_gen;
END_RCPP
}
// ig_entropy
double ig_entropy(double a, double b);
RcppExport SEXP _varapproxr_ig_entropy(SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(ig_entropy(a, b));
    return rcpp_result_gen;
END_RCPP
}
// pnorm_mat
arma::mat pnorm_mat(arma::mat& m);
RcppExport SEXP _varapproxr_pnorm_mat(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(pnorm_mat(m));
    return rcpp_result_gen;
END_RCPP
}
// dnorm_mat
arma::mat dnorm_mat(arma::mat& m);
RcppExport SEXP _varapproxr_dnorm_mat(SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(dnorm_mat(m));
    return rcpp_result_gen;
END_RCPP
}
// vb_lin_reg
List vb_lin_reg(const arma::mat& X, const arma::vec& y, const arma::vec& mu0, const arma::mat& Sigma0, const double a0, const double b0, double tol, int maxiter);
RcppExport SEXP _varapproxr_vb_lin_reg(SEXP XSEXP, SEXP ySEXP, SEXP mu0SEXP, SEXP Sigma0SEXP, SEXP a0SEXP, SEXP b0SEXP, SEXP tolSEXP, SEXP maxiterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma0(Sigma0SEXP);
    Rcpp::traits::input_parameter< const double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< const double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    rcpp_result_gen = Rcpp::wrap(vb_lin_reg(X, y, mu0, Sigma0, a0, b0, tol, maxiter));
    return rcpp_result_gen;
END_RCPP
}
// jaakkola_jordan
double jaakkola_jordan(const arma::mat& X, const arma::vec& y, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p);
RcppExport SEXP _varapproxr_jaakkola_jordan(SEXP XSEXP, SEXP ySEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    rcpp_result_gen = Rcpp::wrap(jaakkola_jordan(X, y, eta1, eta2, eta1_p, eta2_p));
    return rcpp_result_gen;
END_RCPP
}
// jaakkola_jordan_n
double jaakkola_jordan_n(const arma::mat& X, const arma::vec& y, const arma::vec& n, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p);
RcppExport SEXP _varapproxr_jaakkola_jordan_n(SEXP XSEXP, SEXP ySEXP, SEXP nSEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    rcpp_result_gen = Rcpp::wrap(jaakkola_jordan_n(X, y, n, eta1, eta2, eta1_p, eta2_p));
    return rcpp_result_gen;
END_RCPP
}
// saul_jordan
double saul_jordan(const arma::mat& X, const arma::vec& y, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p, arma::vec& omega1);
RcppExport SEXP _varapproxr_saul_jordan(SEXP XSEXP, SEXP ySEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP, SEXP omega1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type omega1(omega1SEXP);
    rcpp_result_gen = Rcpp::wrap(saul_jordan(X, y, eta1, eta2, eta1_p, eta2_p, omega1));
    return rcpp_result_gen;
END_RCPP
}
// saul_jordan_n
double saul_jordan_n(const arma::mat& X, const arma::vec& y, const arma::vec& n, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p, arma::vec& omega1);
RcppExport SEXP _varapproxr_saul_jordan_n(SEXP XSEXP, SEXP ySEXP, SEXP nSEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP, SEXP omega1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type omega1(omega1SEXP);
    rcpp_result_gen = Rcpp::wrap(saul_jordan_n(X, y, n, eta1, eta2, eta1_p, eta2_p, omega1));
    return rcpp_result_gen;
END_RCPP
}
// knowles_minka_wand
double knowles_minka_wand(const arma::mat& X, const arma::vec& y, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p, const arma::vec& MS_p, const arma::vec& MS_s);
RcppExport SEXP _varapproxr_knowles_minka_wand(SEXP XSEXP, SEXP ySEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP, SEXP MS_pSEXP, SEXP MS_sSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type MS_p(MS_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type MS_s(MS_sSEXP);
    rcpp_result_gen = Rcpp::wrap(knowles_minka_wand(X, y, eta1, eta2, eta1_p, eta2_p, MS_p, MS_s));
    return rcpp_result_gen;
END_RCPP
}
// knowles_minka_wand_n
double knowles_minka_wand_n(const arma::mat& X, const arma::vec& y, const arma::vec& n, arma::vec& eta1, arma::vec& eta2, const arma::vec& eta1_p, const arma::vec& eta2_p, const arma::vec& MS_p, const arma::vec& MS_s);
RcppExport SEXP _varapproxr_knowles_minka_wand_n(SEXP XSEXP, SEXP ySEXP, SEXP nSEXP, SEXP eta1SEXP, SEXP eta2SEXP, SEXP eta1_pSEXP, SEXP eta2_pSEXP, SEXP MS_pSEXP, SEXP MS_sSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta1(eta1SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type eta2(eta2SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta1_p(eta1_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type eta2_p(eta2_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type MS_p(MS_pSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type MS_s(MS_sSEXP);
    rcpp_result_gen = Rcpp::wrap(knowles_minka_wand_n(X, y, n, eta1, eta2, eta1_p, eta2_p, MS_p, MS_s));
    return rcpp_result_gen;
END_RCPP
}
// vb_logistic
List vb_logistic(const arma::mat& X, const arma::vec& y, const arma::vec& mu0, const arma::mat& Sigma0, double tol, int maxiter, int maxiter_jj, std::string alg);
RcppExport SEXP _varapproxr_vb_logistic(SEXP XSEXP, SEXP ySEXP, SEXP mu0SEXP, SEXP Sigma0SEXP, SEXP tolSEXP, SEXP maxiterSEXP, SEXP maxiter_jjSEXP, SEXP algSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma0(Sigma0SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter_jj(maxiter_jjSEXP);
    Rcpp::traits::input_parameter< std::string >::type alg(algSEXP);
    rcpp_result_gen = Rcpp::wrap(vb_logistic(X, y, mu0, Sigma0, tol, maxiter, maxiter_jj, alg));
    return rcpp_result_gen;
END_RCPP
}
// vb_logistic_n
List vb_logistic_n(const arma::mat& X, const arma::vec& y, const arma::vec& n, const arma::vec& mu0, const arma::mat& Sigma0, double tol, int maxiter, int maxiter_jj, std::string alg);
RcppExport SEXP _varapproxr_vb_logistic_n(SEXP XSEXP, SEXP ySEXP, SEXP nSEXP, SEXP mu0SEXP, SEXP Sigma0SEXP, SEXP tolSEXP, SEXP maxiterSEXP, SEXP maxiter_jjSEXP, SEXP algSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type n(nSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma0(Sigma0SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter_jj(maxiter_jjSEXP);
    Rcpp::traits::input_parameter< std::string >::type alg(algSEXP);
    rcpp_result_gen = Rcpp::wrap(vb_logistic_n(X, y, n, mu0, Sigma0, tol, maxiter, maxiter_jj, alg));
    return rcpp_result_gen;
END_RCPP
}
// ph_exponential
List ph_exponential(const arma::mat& X, const arma::vec& y, const arma::vec& v, const arma::vec& mu0, const arma::mat& Sigma0, double tol, int maxiter, bool verbose);
RcppExport SEXP _varapproxr_ph_exponential(SEXP XSEXP, SEXP ySEXP, SEXP vSEXP, SEXP mu0SEXP, SEXP Sigma0SEXP, SEXP tolSEXP, SEXP maxiterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type v(vSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Sigma0(Sigma0SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(ph_exponential(X, y, v, mu0, Sigma0, tol, maxiter, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_varapproxr_mvn_entropy", (DL_FUNC) &_varapproxr_mvn_entropy, 1},
    {"_varapproxr_ig_entropy", (DL_FUNC) &_varapproxr_ig_entropy, 2},
    {"_varapproxr_pnorm_mat", (DL_FUNC) &_varapproxr_pnorm_mat, 1},
    {"_varapproxr_dnorm_mat", (DL_FUNC) &_varapproxr_dnorm_mat, 1},
    {"_varapproxr_vb_lin_reg", (DL_FUNC) &_varapproxr_vb_lin_reg, 8},
    {"_varapproxr_jaakkola_jordan", (DL_FUNC) &_varapproxr_jaakkola_jordan, 6},
    {"_varapproxr_jaakkola_jordan_n", (DL_FUNC) &_varapproxr_jaakkola_jordan_n, 7},
    {"_varapproxr_saul_jordan", (DL_FUNC) &_varapproxr_saul_jordan, 7},
    {"_varapproxr_saul_jordan_n", (DL_FUNC) &_varapproxr_saul_jordan_n, 8},
    {"_varapproxr_knowles_minka_wand", (DL_FUNC) &_varapproxr_knowles_minka_wand, 8},
    {"_varapproxr_knowles_minka_wand_n", (DL_FUNC) &_varapproxr_knowles_minka_wand_n, 9},
    {"_varapproxr_vb_logistic", (DL_FUNC) &_varapproxr_vb_logistic, 8},
    {"_varapproxr_vb_logistic_n", (DL_FUNC) &_varapproxr_vb_logistic_n, 9},
    {"_varapproxr_ph_exponential", (DL_FUNC) &_varapproxr_ph_exponential, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_varapproxr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
