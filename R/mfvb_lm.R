mfvb_lm <- function(...) UseMethod("mfvb_lm")

is.mfvb_lm <- function(x) {
  inherits(x, "mfvb_lm")
}

is.mfvb <- function(x) {
  inherits(x, "mfvb")
}


#' Control arguments for mfvb functions
#' 
#' The `control` argument 
#' 
#' @param tol a positive convergence tolerance for ELBO
#' @param maxiter integer giving the maximal number of iterations
#' @param verbose logical indicating if output should be printed for each iteration
#' @return A list with named components
mfvb.control <- function(tol = 1e-8, maxiter = 100, verbose = FALSE) 
{
  if (!is.numeric(tol) || tol <= 0) 
    stop("value of 'tol' must be > 0")
  if (!is.numeric(maxiter) || maxiter <= 0) 
    stop("maximum number of iterations must be > 0")
  list(tol = tol, maxiter = maxiter, verbose = verbose)  
}


#' Check the `x` input to an mfvb regression function
#' 
#' @param x The x input to check
#' @return x if all checks pass
check.mfvb.x <- function(x)
{
  if(!is.matrix(x))
    stop("'x' is not a matrix or could not be converted to a matrix")
  if(NROW(x) == 0 || NCOL(x) == 0)
    stop("'x' is empty")
  if(anyNA(x))
    stop("NA in 'x'")
  if(!is.numeric(x[,1]) && !is.logical(x[,1]))
    stop("non-numeric column in 'x'")
  missing.colnames <- if(is.null(colnames(x))) 1:NCOL(x) else nchar(colnames(x)) == 0
  colnames(x)[missing.colnames] <- c("(Intercept)", paste("V", seq_len(NCOL(x) - 1), sep = ""))[missing.colnames]
  duplicated <- which(duplicated(colnames(x)))
  if(length(duplicated))
    stop("column name \"", colnames(x)[duplicated[1]], "\" in 'x' is duplicated")
  x
}


#' Check the `y` input to an mfvb regression function
#' 
#' @param x The x input
#' @param y The y input to check
#' @return x if all checks pass
check.mfvb.y <- function(x, y)
{
  # as.vector(as.matrix(y)) is necessary when y is a data.frame
  # (because as.vector alone on a data.frame returns a data.frame)
  y <- as.vector(as.matrix(y))
  if(length(y) == 0)
    stop("'y' is empty")
  if(anyNA(y))
    stop("NA in 'y'")
  if(!is.numeric(y) && !is.logical(y))
    stop("'y' is not numeric or logical")
  if(length(y) != nrow(x))
    stop("nrow(x) is ", nrow(x), " but length(y) is ", length(y))
  y
}


#' #' mfvb_lm.fit
#' #' 
#' #' @param x The x input
#' #' @param y The response
#' #' @param weights weighting
#' #' @param subset subset of data
#' #' @param na.action handling of na
#' #' @param offset Offset term
#' #' @param control Control terms
#' #' 
#' #' @return x if all checks pass
#' mfvb_lm.fit <- function(
#'   x = stop("no 'x' argument"),
#'   y = stop("no 'y' argument"),
#'   weights = NULL,
#'   subset = NULL,
#'   na.action = na.fail,
#'   offset = NULL,
#'   control = list(),
#'   ...
#'   ) {
#'   control <- do.call("mfvb.control", control)
#'   x <- check.mfvb.x(x)
#'   y <- check.mfvb.y(x, y)
#' }


#' mfvb_lm
#' 
#' Fit a linear regression model using mean-field variational Bayes approximation.
#' 
#' @param formula an object of class  [stats::formula].
#' @param data an optional data frame, list or environment 
#' (or object coercible by as.data.frame to a data frame) 
#' containing the variables in the model. If not found in data, 
#' the variables are taken from environment(formula), 
#' typically the environment from which lm is called.
#' @param subset subset of data
#' @param weights weighting
#' @param na.action handling of na
#' @param prior a list with elements (mu0, Sigma0, a, b) giving the model prior parameters.
#' @param contrasts design contrasts
#' @param x return design matrix
#' @param y return response vector
#' @param ... further arguments to `vb_lm`
#' @return An `mfvb` object
#' @export
#' @importFrom stats model.response
#' @importFrom stats model.matrix
#' @importFrom stats model.weights
#' @importFrom stats model.offset
#' @importFrom stats setNames
#' @importFrom stats coef
mfvb_lm <- function(
  formula, data, subset, weights, na.action,
  prior = NULL, contrasts = NULL,
  x = FALSE, y = FALSE, ...) {
  
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE) 
  # m <- match(c("formula", "data", "subset", "weights", "na.action", "offset"), names(mf), 0L)
  # mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  
  Y <- model.response(mf, "numeric")
  X <- model.matrix(mt, mf, contrasts)
  W <- as.vector(model.weights(mf))
  offset <- as.vector(model.offset(mf))
  
  if (!is.null(W)) {
    X <- sqrt(W) * X
    Y <- sqrt(W) * Y
  }
  
  if(is.null(prior)) {
    mu0 <- rep(0, ncol(X))
    Sigma0 <- diag(100^2, ncol(X))
    a0 <- 1e-2
    b0 <- 1e-2
    dist <- 1
  } else {
    mu0 <- prior$mu0
    Sigma0 <- prior$Sigma0
    a0 <- prior$a0
    b0 <- prior$b0
    dist <- prior$dist
  }
  
  mfvb_fit <- vb_lm(X, Y, mu0, Sigma0, a0, b0, prior = dist, ...)
  mfvb_fit$call <- mf
  mfvb_fit$mu <- setNames(c(mfvb_fit$mu), colnames(X))
  dimnames(mfvb_fit$Sigma) <- list(colnames(X), colnames(X))
  if(x) mfvb_fit$X <- X
  if(y) mfvb_fit$Y <- Y
  
  class(mfvb_fit) <- "mfvb"
  mfvb_fit
}


#' Extract model coefficients posterior means
#' 
#' `coef.mfvb` implements the `coef` generic for objects of class `mfvb`.
#' 
#' @param object a `mfvb` model
#' @export
coef.mfvb <- function(object) {
  val <- object$mu
  val
}


#' Extract model coefficient posterior covariance matrix
#' 
#' `vcov.mfvb` implements the `vcov` generic for objects of class `mfvb`.
#' 
#' @param object a `mfvb` model
#' @export
vcov.mfvb <- function(object) {
  val <- object$Sigma
  val
}


#' Compute credible intervals for parameters of `mfvb`
#' 
#' @param object a `mfvb` model
#' @param level credible interval level
#' @importFrom stats coef
#' @importFrom stats vcov
confint.mfvb <- function(object, level = 0.95) {
  cf <- coef(object)
  sds <- sqrt(diag(vcov(object)))
  pnames <- names(sds)
  if (is.matrix(cf)) 
    cf <- stats::setNames(as.vector(cf), pnames)
  a <- (1 - level) / 2
  a <- c(a, 1 - a)
  z <- stats::qnorm(a)
  ci <- cf + sds %o% z
  colnames(ci) <- sprintf("%2.1f%%", 100*a)
  ci
}


#' Summarise MFVB fit
#' 
#' @param object an object of class `mfvb`
#' @export
summary.mfvb <- function(object) {
  beta <- coef(object)
  V <- vcov(object)
  S <- sqrt(diag(V))
  P <- 1 - stats::pnorm(0, beta, S)
  sTable <- cbind("mean" = beta, "SD" = S, "Pr(>0)" = P)
  printCoefmat(sTable)
}


#' Sample from MFVB fit
#' 
#' @param object an object of class `mfvb`
#' @param par a character vector indicating which distributions to sample
#' @return A matrix of draws from the indicated distributions
#' @export
sample_vbdist <- function(object, par = NULL, n_samples = 1e3) {
  if(is.null(par)) {
    par <- c("mu", "sigma")
  }
  out <- NULL
  if("mu" %in% par) {
    mu <- coef(fit) 
    Sigma <- vcov(fit)
    beta_draws <- mvnfast::rmvn(n_samples, mu, Sigma)
    colnames(beta_draws) <- names(mu)
    out$beta <- beta_draws
  }
  if("sigma" %in% par) {
    a <- fit$a
    b <- fit$b
    sigma_draws <- matrix(sqrt(1 / rgamma(n_samples, a, b)), n_samples, 1)
    colnames(sigma_draws) <- "sigma"
    out$sigma <- sigma_draws
  }
  return(do.call(cbind, out))
}
