# Test Inverse-Gamma functions

test_that("Inverse Gamma functions work as expected", {
  expect_true(is.na(ig_E(2, -2)))
  expect_true(is.na(ig_E(1, 2)))
  expect_equal(ig_E(2, 2), 2)
  expect_equal(ig_E_inv(1, 2), 0.5)
  expect_equal(ig_E_log(1, 2), log(2) - digamma(1))
})

test_that("Inverse Wishart functions work as expected", {
  expect_error(inv_wishart_E_invX(2, diag(1,2)))
  expect_error(inv_wishart_E_invX(3, matrix(c(1,2,3,4), 2, 2)))
  expect_equal(inv_wishart_E_invX(4, diag(1, 2)), diag(4, 2))
  
  expect_error(inv_wishart_E_logdet(2, diag(1,2)))
  expect_error(inv_wishart_E_logdet(3, matrix(c(1,2,3,4), 2, 2)))
  expect_equal(inv_wishart_E_logdet(2, diag(1, 1)), log(0.5) + digamma(2))
  expect_equal(inv_wishart_E_logdet(3, diag(1, 2)), 2*log(0.5) + digamma(3) + digamma(2.5))
  expect_equal(inv_wishart_E_logdet(4, diag(2, 2)), 2*log(0.5) + digamma(4) + digamma(3.5) + log(det(diag(2, 2))))
})