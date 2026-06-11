context("Test implementation of gower_mixed distance ...")

test_that("gower_mixed computes the correct distance for mixed numeric and categorical data.", {
  # Create a data frame with numeric and categorical features
  x <- data.frame(
    num1 = c(1, 3),
    num2 = c(2, 4),
    cat1 = as.integer(c(1, 1)),
    cat2 = as.integer(c(2, 3))
  )

  # Expected distances for pair (1,2):
  # Numeric: |1-3|/2 + |2-4|/2 = 1 + 1 = 2 (ranges are 2 each: 1-3 and 2-4)
  # Categorical: (1==1 ? 0 : 1) + (2==3 ? 0 : 1) = 0 + 1 = 1
  # Total dist = 2 + 1 = 3; total weight = 4
  # Result: 3/4 = 0.75

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 0.75)
})

test_that("gower_mixed handles weighting correctly.", {
  x <- data.frame(
    num1 = c(1, 3),
    num2 = c(2, 4),
    cat1 = as.integer(c(1, 1)),
    cat2 = as.integer(c(2, 3))
  )

  # With weights [2, 1, 3, 1] for [num1, num2, cat1, cat2]
  # Numeric: 2 * (|1-3|/2) + 1 * (|2-4|/2) = 2*1 + 1*1 = 3
  # Categorical: 3 * 0 + 1 * 1 = 1
  # Total dist = 3 + 1 = 4; total weight = 2 + 1 + 3 + 1 = 7
  # Result: 4/7

  weights <- c(2, 1, 3, 1)
  res <- philentropy::gower_mixed(x, weights = weights)
  expect_equal(res[1, 2], 4/7)
})

test_that("gower_mixed handles NA values correctly.", {
  x <- data.frame(
    num1 = c(1, 3),
    num2 = c(NA, 4),
    cat1 = as.integer(c(1, 1)),
    cat2 = as.integer(c(2, NA))
  )

  # For pair (1,2):
  # num1: (1 vs 3) valid, |1-3|/2 * 1 = 1, weight = 1
  # num2: (NA vs 4) -> NA excluded, no contribution, weight = 0
  # cat1: (1 vs 1) valid, 0 * 1 = 0, weight = 1
  # cat2: (2 vs NA) -> NA excluded, no contribution, weight = 0
  # Total dist = 1 + 0 = 1; total weight = 1 + 0 + 1 + 0 = 2
  # Result: 1/2 = 0.5

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 0.5)
})

# Test Fix 3: Logical column handling
test_that("gower_mixed converts logical columns to 0/1 numeric.", {
  x <- data.frame(
    flag = c(TRUE, FALSE),
    val = c(1, 2)
  )
  # Both columns are now numeric: logical TRUE=1, FALSE=0 and val as-is
  # Distance between (1, 1) and (0, 2):
  # flag: |1-0|/1 = 1 (range is 1-0=1)
  # val: |1-2|/1 = 1 (range is 2-1=1)
  # Total = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test Fix 2: C++ range computation produces consistent results
test_that("gower_mixed with pre-supplied ranges matches computed ranges.", {
  x <- data.frame(
    num1 = c(1, 3),
    num2 = c(2, 4),
    cat1 = factor(c("a", "a")),
    cat2 = factor(c("x", "y"))
  )

  # Compute ranges and use them explicitly
  ranges_computed <- c(2, 2)  # 3-1=2, 4-2=2 for numeric columns
  res_with_ranges <- philentropy::gower_mixed(x, ranges = ranges_computed)
  res_without_ranges <- philentropy::gower_mixed(x)

  # Results should match
  expect_equal(res_with_ranges, res_without_ranges)
})

# Test data type routing: factor columns
test_that("gower_mixed handles unordered factor columns correctly.", {
  x <- data.frame(
    num = c(1, 3),
    fac = factor(c("a", "b"))
  )

  # num: |1-3|/2 = 1 (range 3-1=2)
  # fac: 1 (different levels)
  # Total dist = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test data type routing: character columns
test_that("gower_mixed handles character columns correctly.", {
  x <- data.frame(
    num = c(1, 3),
    chr = c("a", "b")
  )

  # num: |1-3|/2 = 1 (range 3-1=2)
  # chr: 1 (different strings, treated as categorical)
  # Total dist = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test data type routing: ordered factor columns (converted to numeric)
test_that("gower_mixed handles ordered factor columns as numeric.", {
  x <- data.frame(
    ord_fac = ordered(c("low", "high"), levels = c("low", "mid", "high")),
    val = c(1, 3)
  )
  # Ordered factors are converted to 1, 3 (level indices)
  # Both columns now numeric: |1-3|/2 = 1 for both
  # Total dist = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test all-numeric input
test_that("gower_mixed works with all-numeric data.", {
  x <- data.frame(
    a = c(1, 5),
    b = c(2, 6)
  )
  # Both columns treated as numeric
  # a: |1-5|/4 = 1 (range 5-1=4)
  # b: |2-6|/4 = 1 (range 6-2=4)
  # Total dist = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test all-categorical input
test_that("gower_mixed works with all-categorical data.", {
  x <- data.frame(
    fac1 = factor(c("a", "b")),
    fac2 = factor(c("x", "y"))
  )
  # Both columns treated as categorical
  # fac1: 1 (different)
  # fac2: 1 (different)
  # Total dist = 2, weight = 2, result = 1.0

  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 1.0)
})

# Test cross-distance (y parameter)
test_that("gower_mixed with y parameter computes cross-distance matrix.", {
  x <- data.frame(
    num = c(1, 3),
    cat = as.integer(c(1, 1))
  )
  y <- data.frame(
    num = c(5),
    cat = as.integer(c(2))
  )

  res <- philentropy::gower_mixed(x, y = y)
  expect_equal(dim(res), c(2, 1))  # 2x1 matrix

  # Check first row distance
  # row 1 (1, 1) vs y (5, 2):
  # num: |1-5|/4 = 1 (range over both datasets: 5-1=4)
  # cat: 1 (1 != 2)
  # Total = 2, weight = 2, result = 1.0
  expect_equal(res[1, 1], 1.0)
})

# Test range validation
test_that("gower_mixed validates ranges length.", {
  x <- data.frame(
    num1 = c(1, 3),
    num2 = c(2, 4),
    cat = as.integer(c(1, 1))
  )

  # Wrong ranges length: only 1 instead of 2 numeric columns
  expect_error(
    philentropy::gower_mixed(x, ranges = c(2)),
    "does not match"
  )
})

# Test weights validation
test_that("gower_mixed validates weights length.", {
  x <- data.frame(
    num = c(1, 3),
    cat = as.integer(c(1, 1))
  )

  # Wrong weights length: 1 instead of 2 total columns
  expect_error(
    philentropy::gower_mixed(x, weights = c(1)),
    "must have length equal"
  )
})

# Test Fix 4: distance() parameter validation
test_that("distance() rejects weights/ranges for non-gower methods.", {
  x <- matrix(1:6, nrow = 2)

  expect_error(
    philentropy::distance(x, method = "euclidean", weights = c(1, 1, 1)),
    "weights.*only supported"
  )

  expect_error(
    philentropy::distance(x, method = "manhattan", ranges = c(1, 1, 1)),
    "ranges.*only supported"
  )
})

# Test Fix 4: distance() with gower_mixed routing
test_that("distance() routes mixed data to gower_mixed correctly.", {
  x <- data.frame(
    num = c(1, 3),
    cat = factor(c("a", "b"))
  )

  res <- philentropy::distance(x, method = "gower_mixed")
  expect_is(res, "matrix")
  expect_equal(dim(res), c(2, 2))
})

# Test Fix 4: distance() with gower_numeric extracts numeric columns and routes to gower_mixed
test_that("distance() with gower_numeric routes through gower_mixed with numeric subset.", {
  x <- data.frame(
    num1 = c(1, 5),
    num2 = c(2, 6),
    cat = factor(c("a", "b"))
  )

  res <- philentropy::distance(x, method = "gower_numeric")

  # Should compute gower on numeric columns only
  # num1: |1-5|/4 = 1
  # num2: |2-6|/4 = 1
  # Total = 2, weight = 2, result = 1.0
  expect_equal(res[1, 2], 1.0)
})

# Test Fix 4: distance() with gower_numeric forwards weights to numeric subset
test_that("distance() with gower_numeric forwards weights correctly.", {
  x <- data.frame(
    num = c(1, 5),
    cat = factor(c("a", "b"))
  )

  # Weight only the numeric column (weight 2)
  res <- philentropy::distance(x, method = "gower_numeric", weights = c(2))

  # num: (|1-5|/4) * 2 = 2
  # Total dist = 2, weight = 2, result = 1.0
  expect_equal(res[1, 2], 1.0)
})

# Test edge case: NA in all rows of a column
test_that("gower_mixed handles all-NA columns.", {
  x <- data.frame(
    num1 = c(NA, NA),
    num2 = c(1, 2),
    cat = as.integer(c(1, 1))
  )

  # num1 contributes nothing (all NA)
  # num2: |1-2|/1 = 1 (range 2-1=1)
  # cat: 0 (same)
  # Total dist = 1, weight = 2 (num2 + cat, num1 excluded), result = 0.5
  res <- philentropy::gower_mixed(x)
  expect_equal(res[1, 2], 0.5)
})
