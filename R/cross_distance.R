#  Part of the philentropy package
#
#  Copyright (C) 2015-2026 Hajk-Georg Drost
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

#' @title Compute Distance Between Two Datasets with Aggregation
#' @description Computes cross-dataset distances between all pairs of rows from two matrices or data.frames,
#' and optionally aggregates the resulting distance matrix into a single scalar value. Automatically routes
#' mixed-type data (numeric + categorical) to Gower distance, and numeric-only data to the specified distance metric.
#' @param x a \code{data.frame} or \code{matrix} containing features (rows = samples, columns = features).
#' @param y a \code{data.frame} or \code{matrix} with the same number of columns as \code{x}.
#' @param method a character string specifying the distance metric. Default is \code{"euclidean"}. For data
#' with categorical columns, \code{"gower_mixed"} or \code{"gower"} is recommended. All methods supported by
#' \code{\link{distance}} are available.
#' @param aggregate an aggregation function or character string specifying how to combine the \code{nrow(x)} by \code{nrow(y)}
#' cross-distance matrix into a single value. Options:
#' \itemize{
#'   \item \code{"mean"} (default): mean of all pairwise distances
#'   \item \code{"median"}: median of all pairwise distances
#'   \item \code{"min"}: minimum distance
#'   \item \code{"max"}: maximum distance
#'   \item \code{"hausdorff"}: symmetric Hausdorff distance: \code{max(max(apply(D, 1, min)), max(apply(D, 2, min)))}
#'   \item \code{NULL}: return the full \code{nrow(x)} by \code{nrow(y)} distance matrix without aggregation
#'   \item custom function: any function that takes a matrix and returns a scalar
#' }
#' @param weights optional numeric vector of feature weights. Length must equal \code{ncol(x)}. Only used with Gower distance.
#' @param ranges optional numeric vector of custom ranges for numeric features. Only used with Gower distance.
#' @param num_threads integer specifying the number of threads for parallel computation. Default: 2.
#' @param ... additional arguments passed to the underlying distance function.
#' @return If \code{aggregate = NULL}, returns an \code{nrow(x)} by \code{nrow(y)} \code{matrix}. Otherwise, returns a scalar
#' (numeric) aggregating the distances.
#' @details
#' The function automatically detects mixed-type data (factors or characters in addition to numeric columns)
#' and routes to \code{\link{gower_mixed}} for proper handling. For purely numeric data, it routes to
#' \code{\link{dist_many_many}} with the specified method.
#'
#' The Hausdorff distance is computed as the maximum of:
#' \itemize{
#'   \item maximum of the minimum distances from each row of x to any row of y
#'   \item maximum of the minimum distances from each row of y to any row of x
#' }
#' This is useful for assessing how well one dataset "covers" another.
#' @seealso \code{\link{gower_mixed}}, \code{\link{dist_many_many}}, \code{\link{distance}}
#' @examples
#' \dontrun{
#' # Numeric datasets
#' M1 <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
#' M2 <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 2)
#' cross_distance(M1, M2, method = "euclidean")              # scalar (mean distance)
#' cross_distance(M1, M2, method = "euclidean", aggregate = "hausdorff")
#' cross_distance(M1, M2, aggregate = NULL)                 # full 2 by 2 distance matrix
#'
#' # Mixed-type data
#' x <- data.frame(age = c(25, 30, 35), education = factor(c(1, 2, 2)))
#' y <- data.frame(age = c(28, 40), education = factor(c(1, 3)))
#' cross_distance(x, y, method = "gower_mixed")
#'
#' # Custom aggregation
#' cross_distance(M1, M2, method = "euclidean", aggregate = function(D) median(D))
#' }
#' @export
cross_distance <- function(x, y, method = "euclidean", aggregate = "mean",
                            weights = NULL, ranges = NULL, num_threads = 2L, ...) {
  if (is.matrix(x)) {
    x <- as.data.frame(x)
  } else if (!is.data.frame(x)) {
    stop("x must be a data.frame or matrix")
  }

  if (is.matrix(y)) {
    y <- as.data.frame(y)
  } else if (!is.data.frame(y)) {
    stop("y must be a data.frame or matrix")
  }

  if (ncol(x) != ncol(y)) {
    stop("x and y must have the same number of columns")
  }

  if (.is_mixed_df(x) || .is_mixed_df(y) || method %in% c("gower_mixed", "gower")) {
    dist_matrix <- gower_mixed(x, y = y, weights = weights, ranges = ranges,
                               num_threads = num_threads)
  } else {
    x_mat <- as.matrix(x)
    y_mat <- as.matrix(y)

    if (is.logical(x_mat)) storage.mode(x_mat) <- "double"
    if (is.logical(y_mat)) storage.mode(y_mat) <- "double"

    if (!is.numeric(x_mat)) {
      stop("After converting to matrix, x contains non-numeric values. Use method = 'gower_mixed' for mixed data.")
    }
    if (!is.numeric(y_mat)) {
      stop("After converting to matrix, y contains non-numeric values. Use method = 'gower_mixed' for mixed data.")
    }

    dist_matrix <- dist_many_many(x_mat, y_mat, method = method,
                                  num_threads = num_threads, ranges = ranges, ...)
  }

  if (is.null(aggregate)) {
    return(dist_matrix)
  }

  if (is.character(aggregate)) {
    aggregate <- switch(aggregate,
      "mean" = function(D) mean(D),
      "median" = function(D) median(D),
      "min" = function(D) min(D),
      "max" = function(D) max(D),
      "hausdorff" = function(D) max(max(apply(D, 1, min)), max(apply(D, 2, min))),
      stop(sprintf("Unknown aggregation method: %s. Use 'mean', 'median', 'min', 'max', 'hausdorff', NULL, or a custom function.", aggregate))
    )
  } else if (!is.function(aggregate)) {
    stop("aggregate must be a character string, NULL, or a function")
  }

  aggregate(dist_matrix)
}
