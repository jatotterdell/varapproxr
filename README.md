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

# Dependencies

varapproxr uses:

- [GCEM](https://github.com/kthohr/gcem) for calculation of some special mathematics functions.
- [StatsLib](https://github.com/kthohr/stats) for functions of probability distributions.
