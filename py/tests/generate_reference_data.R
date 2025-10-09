#!/usr/bin/env Rscript
# Generate reference CSV data for Python tests
# Run from py/tests directory: Rscript generate_reference_data.R

library(philentropy)

output_dir <- "data"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Generating reference data for Python tests...\n")

# Test data 1: Basic euclidean distance
x1 <- rbind(
  1:5 / sum(1:5),
  5:1 / sum(5:1),
  rep(1, 5) / 5
)
result1 <- distance(x1, method = "euclidean", mute.message = TRUE)
result1_clean <- unname(as.matrix(result1))
write.table(result1_clean, file.path(output_dir, "distance_euclidean.csv"), 
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat("Generated: distance_euclidean.csv\n")

# Test data 2: Kullback-Leibler divergence
x2 <- rbind(
  c(0.25, 0.25, 0.25, 0.25),
  c(0.1, 0.2, 0.3, 0.4),
  c(0.4, 0.3, 0.2, 0.1)
)
result2 <- distance(x2, method = "kullback-leibler", mute.message = TRUE)
result2_clean <- unname(as.matrix(result2))
write.table(result2_clean, file.path(output_dir, "distance_kullback.csv"), 
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat("Generated: distance_kullback.csv\n")

# Test data 3: Empirical probability estimation
x3 <- rbind(
  c(10, 20, 30, 40),
  c(40, 30, 20, 10),
  c(25, 25, 25, 25)
)
result3 <- distance(x3, method = "euclidean", est.prob = "empirical", mute.message = TRUE)
result3_clean <- unname(as.matrix(result3))
write.table(result3_clean, file.path(output_dir, "distance_counts_empirical.csv"), 
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat("Generated: distance_counts_empirical.csv\n")

# Test data 4: One-to-many Manhattan distance
ref <- 1:5 / sum(1:5)
others <- rbind(
  5:1 / sum(5:1),
  rep(1, 5) / 5,
  rep(2, 5) / 10
)
result4 <- dist_one_many(ref, others, method = "manhattan")
result4_clean <- unname(as.matrix(result4))
write.table(result4_clean, file.path(output_dir, "dist_one_many_manhattan.csv"), 
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
cat("Generated: dist_one_many_manhattan.csv\n")

# Test data 5: Hellinger distance (single value)
p <- 1:10 / sum(1:10)
q <- 20:29 / sum(20:29)
result5 <- dist_one_one(p, q, method = "hellinger")
writeLines(as.character(result5), file.path(output_dir, "dist_one_one_hellinger.txt"))
cat("Generated: dist_one_one_hellinger.txt\n")

cat("\nAll reference data generated successfully!\n")
cat("Run Python tests with: cd py && python -m pytest tests/ -v\n")
