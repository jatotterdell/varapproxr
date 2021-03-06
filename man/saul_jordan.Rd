% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{saul_jordan}
\alias{saul_jordan}
\title{Perform Saul-Jordan update of variational parameters}
\usage{
saul_jordan(X, y, eta1, eta2, eta1_p, eta2_p, omega1)
}
\arguments{
\item{X}{The design matrix}

\item{y}{The response vector}

\item{eta1}{The current value of 1st natural parameter}

\item{eta2}{The current value of the 2nd natural parameter}

\item{eta1_p}{The prior value of the 1st natural parameter}

\item{eta2_p}{The prior value of the 2nd natural parameter}

\item{omega1}{The current value of the Omega1 variational parameter}
}
\description{
Perform Saul-Jordan update of variational parameters
}
