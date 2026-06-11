context("Test cross_distance function ...")

test_that("cross_distance returns full matrix when aggregate = NULL", {
  M1 <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  M2 <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 2)

  result <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)

  expect_is(result, "matrix")
  expect_equal(dim(result), c(2, 2))
  expect_true(all(is.numeric(result)))
})

test_that("cross_distance with mean aggregation equals manual mean", {
  M1 <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
  M2 <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 2)

  full_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)
  mean_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = "mean")

  expect_equal(mean_dist, mean(full_dist))
})

test_that("cross_distance with median aggregation equals manual median", {
  M1 <- matrix(runif(20), nrow = 4)
  M2 <- matrix(runif(30), nrow = 6)

  full_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)
  median_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = "median")

  expect_equal(median_dist, median(full_dist))
})

test_that("cross_distance with min/max aggregation works", {
  M1 <- matrix(runif(20), nrow = 4)
  M2 <- matrix(runif(30), nrow = 6)

  full_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)
  min_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = "min")
  max_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = "max")

  expect_equal(min_dist, min(full_dist))
  expect_equal(max_dist, max(full_dist))
})

test_that("cross_distance with hausdorff aggregation is correct", {
  M1 <- matrix(c(1, 2, 3, 4), nrow = 2)
  M2 <- matrix(c(5, 6, 7, 8), nrow = 2)

  full_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)
  hausdorff_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = "hausdorff")

  expected <- max(max(apply(full_dist, 1, min)), max(apply(full_dist, 2, min)))
  expect_equal(hausdorff_dist, expected)
})

test_that("cross_distance with custom function works", {
  M1 <- matrix(c(1, 2, 3, 4), nrow = 2)
  M2 <- matrix(c(5, 6, 7, 8), nrow = 2)

  full_dist <- cross_distance(M1, M2, method = "euclidean", aggregate = NULL)
  custom_dist <- cross_distance(M1, M2, method = "euclidean",
                                aggregate = function(D) quantile(D, 0.75))

  expect_equal(custom_dist, quantile(full_dist, 0.75))
})

test_that("cross_distance routes mixed data to gower_mixed", {
  x <- data.frame(
    num1 = c(1, 2),
    cat1 = factor(c("a", "b"))
  )
  y <- data.frame(
    num1 = c(3, 4),
    cat1 = factor(c("a", "c"))
  )

  result <- cross_distance(x, y, method = "gower_mixed")
  expect_is(result, "numeric")
  expect_length(result, 1)
  expect_true(result >= 0)
})

test_that("cross_distance with factors routes to gower_mixed automatically", {
  x <- data.frame(
    age = c(25, 30),
    education = factor(c(1, 2))
  )
  y <- data.frame(
    age = c(28, 35),
    education = factor(c(2, 2))
  )

  result <- cross_distance(x, y, method = "euclidean")
  expect_is(result, "numeric")
  expect_length(result, 1)
})

test_that("cross_distance validates column count mismatch", {
  M1 <- matrix(1:6, nrow = 2)  # 2x3
  M2 <- matrix(1:12, nrow = 3)  # 3x4

  expect_error(cross_distance(M1, M2), "same number of columns")
})

test_that("cross_distance works with data.frame input", {
  x <- data.frame(a = c(1, 2, 3), b = c(4, 5, 6))
  y <- data.frame(a = c(7, 8), b = c(9, 10))

  result <- cross_distance(x, y, method = "euclidean", aggregate = "mean")
  expect_is(result, "numeric")
  expect_length(result, 1)
})

test_that("cross_distance with manhattan method works", {
  M1 <- matrix(c(1, 2, 3, 4), nrow = 2)
  M2 <- matrix(c(5, 6, 7, 8), nrow = 2)

  result <- cross_distance(M1, M2, method = "manhattan", aggregate = "mean")
  expect_is(result, "numeric")
  expect_length(result, 1)
})

test_that("cross_distance recognizes invalid aggregation method", {
  M1 <- matrix(c(1, 2, 3, 4), nrow = 2)
  M2 <- matrix(c(5, 6, 7, 8), nrow = 2)

  expect_error(
    cross_distance(M1, M2, method = "euclidean", aggregate = "invalid"),
    "Unknown aggregation method"
  )
})

test_that("cross_distance returns different dimensions for different inputs", {
  M1 <- matrix(runif(12), nrow = 3)
  M2_2rows <- matrix(runif(8), nrow = 2)
  M2_4rows <- matrix(runif(16), nrow = 4)

  result1 <- cross_distance(M1, M2_2rows, aggregate = NULL)
  result2 <- cross_distance(M1, M2_4rows, aggregate = NULL)

  expect_equal(dim(result1), c(3, 2))
  expect_equal(dim(result2), c(3, 4))
})

test_that("cross_distance with gower_mixed method explicitly works", {
  x <- data.frame(a = c(1, 2), b = c(3, 4))
  y <- data.frame(a = c(5, 6), b = c(7, 8))

  result <- cross_distance(x, y, method = "gower_mixed", aggregate = "mean")
  expect_is(result, "numeric")
  expect_length(result, 1)
})
