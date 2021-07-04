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
#' @param prior a list with elements (mu0, Sigma0, a, b) giving the model prior parameters.
#' @return
#' @export
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
coef.mfvb <- function(object, ...) {
  val <- object$mu
  val
}


#' Extract model coefficient posterior covariance matrix
#' 
#' `vcov.mfvb` implements the `vcov` generic for objects of class `mfvb`.
#' 
#' @param object a `mfvb` model
vcov.mfvb <- function(object, ...) {
  val <- object$Sigma
  val
}


#' Compute credible intervals for parameters of `mfvb`
#' 
#' @param object a `mfvb` model
#' @param level credible interval level
confint.mfvb <- function(object, level = 0.95, ...) {
  cf <- coef(object)
  sds <- sqrt(diag(vcov(object)))
  pnames <- names(sds)
  if (is.matrix(cf)) 
    cf <- setNames(as.vector(cf), pnames)
  a <- (1 - level) / 2
  a <- c(a, 1 - a)
  z <- qnorm(a)
  ci <- cf + sds %o% z
  colnames(ci) <- sprintf("%2.1f%%", 100*a)
  ci
}


#' Summarise MFVB fit
#' 
#' @param object an object of class `mfvb`
#' @param ... further arguments
summary.mfvb <- function(object, ...) {
  beta <- coef(object)
  V <- vcov(object)
  S <- sqrt(diag(V))
  P <- 1 - pnorm(0, beta, S)
  sTable <- data.frame(beta, S, P)
  dimnames(sTable) <- list(names(beta), c("mean", "SD", "Pr(>0)"))
  sTable
}
