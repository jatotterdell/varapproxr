# Test helper functions

test_that("Special mathematics functions work as expected", {
  
  expect_error(lmvgamma(-1, 2))
  expect_error(lmvgamma(2, 0))
  expect_equal(lmvgamma(2, 1), 0)
  expect_equal(lmvgamma(3, 1), 0.693147, tolerance = 1e-5)
  expect_equal(lmvgamma(3, 2), 1.550195, tolerance = 1e-5)
  expect_equal(lmvgamma(4, 3), 5.402975, tolerance = 1e-5)
  
  expect_error(mvdigamma(-1, 2))
  expect_error(mvdigamma(2, 0))
  expect_equal(mvdigamma(2, 1), digamma(2))
  expect_equal(mvdigamma(2, 2), digamma(2) + digamma(1.5))
  expect_equal(mvdigamma(4, 3), 3.282059, tolerance = 1e-5)
  
})