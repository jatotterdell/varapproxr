# varapproxr

<!-- badges: start -->
[![R-CMD-check](https://github.com/jatotterdell/varapproxr/workflows/R-CMD-check/badge.svg)](https://github.com/jatotterdell/varapproxr/actions)
<!-- badges: end -->

This is a work in progress repository for implementations of variational approximate inference.
All implementations are currently prototypes; accuracy and efficiency is not guaranteed.

# Installation

```
# install.packages("devtools")
devtools::install_github('jatotterdell/varapproxr')
```

# Possible Features

- [ ] Optimise C++ functions for numerical stability
- [ ] Write wrappers to interface with standard S3 regression functions
- [ ] Integrate with [`distr`](https://alan-turing-institute.github.io/distr6/) package for OO priors/posteriors
