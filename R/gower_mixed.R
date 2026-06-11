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

.convert_ordered_factors <- function(df) {
  df[] <- lapply(df, function(x) {
    if (is.ordered(x)) {
      x <- as.integer(x)
    }
    x
  })
  df
}

.convert_categorical_to_integer <- function(cat_df) {
  as.matrix(data.frame(lapply(cat_df, function(col) {
    if (!is.integer(col) && !is.factor(col)) {
      col <- factor(col)
    }
    as.integer(col)
  })))
}

#' @title Gower Distance for Mixed Data Types
#' @description Computes pairwise Gower distances between samples with mixed numeric and categorical features.
#' If \code{y} is \code{NULL}, computes pairwise distances within \code{x}. Otherwise, computes distances between rows of \code{x} and rows of \code{y}.
#' @param x a \code{data.frame} or \code{matrix} containing numeric, logical, and/or categorical (integer/factor/character) columns.
#' @param y optional; a second \code{data.frame} or \code{matrix} with the same columns as \code{x}. If provided, distances are computed between rows of \code{x} and rows of \code{y}.
#' @param weights optional numeric vector of weights for each feature. If \code{NULL} (default), all features are weighted equally. Must have length equal to \code{ncol(x)}.
#' @param ranges optional numeric vector of pre-computed ranges for numeric columns. If \code{NULL} (default), ranges are computed from \code{x} (and \code{y} if provided).
#' @param num_threads integer specifying the number of threads for parallel computation. Default: 2.
#' @return A numeric distance matrix of size \code{nrow(x)} by \code{nrow(x)} if \code{y} is \code{NULL}, or \code{nrow(x)} by \code{nrow(y)} if \code{y} is provided.
#' @details
#' Gower distance is a metric for mixed data that combines:
#' - For numeric features: normalized absolute difference divided by range
#' - For logical features: automatically converted to integer 0/1 and treated as numeric features
#' - For ordered factors: automatically converted to numeric positions and treated as numeric features,
#'   allowing distances to respect the ordinal structure (e.g., distance from
#'   "low" to "medium" is less than "low" to "high")
#' - For character and unordered factor columns: 0 if equal, 1 if different
#' - Missing values (\code{NA}) are excluded from the computation
#' @references Gower, J. C. (1971). A general coefficient of similarity and some of its properties. _Biometrics_, 27(4), 857-871. \doi{https://doi.org/10.2307/2528823}
#' @examples
#' \dontrun{
#' x <- data.frame(
#'   age = c(25, 30, 35),
#'   income = c(50000, 60000, 55000),
#'   education = c(1, 2, 2)  # 1=HS, 2=College
#' )
#' gower_mixed(x)
#' }
#' @export
gower_mixed <- function(x, y = NULL, weights = NULL, ranges = NULL, num_threads = 2L) {
  if (is.matrix(x)) {
    x <- as.data.frame(x)
  } else if (!is.data.frame(x)) {
    stop("x must be a data.frame or matrix")
  }

  if (!is.null(y)) {
    if (is.matrix(y)) {
      y <- as.data.frame(y)
    } else if (!is.data.frame(y)) {
      stop("y must be a data.frame or matrix")
    }
    if (ncol(x) != ncol(y)) {
      stop("x and y must have the same number of columns")
    }
  }

  x <- .convert_ordered_factors(x)
  if (!is.null(y)) {
    y <- .convert_ordered_factors(y)
  }

  # Convert logical columns to integer (0/1) so they're treated as numeric
  x[] <- lapply(x, function(col) if (is.logical(col)) as.integer(col) else col)
  if (!is.null(y)) {
    y[] <- lapply(y, function(col) if (is.logical(col)) as.integer(col) else col)
  }

  n <- nrow(x)
  p <- ncol(x)

  is_numeric <- sapply(x, function(col) is.numeric(col))
  numeric_cols <- which(is_numeric)
  categorical_cols <- which(!is_numeric)

  if (length(numeric_cols) > 0) {
    x_num <- as.matrix(x[, numeric_cols, drop = FALSE])
  } else {
    x_num <- matrix(numeric(0), nrow = n, ncol = 0)
  }

  if (length(categorical_cols) > 0) {
    x_cat <- as.data.frame(x[, categorical_cols, drop = FALSE])
    x_cat <- .convert_categorical_to_integer(x_cat)
  } else {
    x_cat <- matrix(integer(0), nrow = n, ncol = 0)
  }

  if (is.null(ranges)) {
    if (length(numeric_cols) > 0) {
      if (!is.null(y)) {
        y_num <- as.matrix(y[, numeric_cols, drop = FALSE])
        ranges <- compute_gower_ranges_cross_cpp(x_num, y_num)
      } else {
        ranges <- compute_gower_ranges_cpp(x_num)
      }
    } else {
      ranges <- numeric(0)
    }
  } else {
    if (length(ranges) != length(numeric_cols)) {
      stop("'ranges' length (", length(ranges), ") does not match number of numeric columns (",
           length(numeric_cols), ")")
    }
  }

  if (is.null(weights)) {
    w_num <- rep(1, length(numeric_cols))
    w_cat <- rep(1, length(categorical_cols))
  } else {
    if (length(weights) != p) {
      stop("weights must have length equal to ncol(x)")
    }
    w_num <- weights[numeric_cols]
    w_cat <- weights[categorical_cols]
  }
  if (is.null(y)) {
    result <- gower_mixed_cpp(
      x_num = x_num,
      x_cat = x_cat,
      ranges = ranges,
      w_num = w_num,
      w_cat = w_cat,
      num_threads = as.integer(num_threads)
    )
    rownames(result) <- rownames(x)
    colnames(result) <- rownames(x)
  } else {
    if (length(numeric_cols) > 0) {
      y_num <- as.matrix(y[, numeric_cols, drop = FALSE])
    } else {
      y_num <- matrix(numeric(0), nrow = nrow(y), ncol = 0)
    }

    if (length(categorical_cols) > 0) {
      y_cat <- as.data.frame(y[, categorical_cols, drop = FALSE])
      y_cat <- .convert_categorical_to_integer(y_cat)
    } else {
      y_cat <- matrix(integer(0), nrow = nrow(y), ncol = 0)
    }

    result <- gower_cross_cpp(
      x_num = x_num,
      x_cat = x_cat,
      y_num = y_num,
      y_cat = y_cat,
      ranges = ranges,
      w_num = w_num,
      w_cat = w_cat,
      num_threads = as.integer(num_threads)
    )
    rownames(result) <- rownames(x)
    colnames(result) <- rownames(y)
  }

  return(result)
}
