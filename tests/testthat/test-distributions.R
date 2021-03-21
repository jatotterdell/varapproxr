# Test Inverse-Gamma functions

test_that("E[1/x] is correct for x ~ IG(a,b)", {
  expect_equal(ig_E_inv(1, 2), 0.5)
})
test_that("E[ln(x)] is correct for x ~ IG(a,b)", {
  expect_equal(ig_E_log(1, 2), log(2) - digamma(1))
})