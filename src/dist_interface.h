#ifndef PHILENTROPY_DIST_INTERFACE_H
#define PHILENTROPY_DIST_INTERFACE_H

#include <cstddef>
#include <string>
#include "common_utils.h"

namespace philentropy {

struct DistConfig {
    std::string method;
    std::string unit;
    double epsilon;
    double p;
    bool has_p;
};

enum class DistributionAxis {
    Rows,
    Columns
};

struct MatrixView {
    const double* data;
    std::size_t rows;
    std::size_t cols;
    std::size_t row_stride;
    std::size_t col_stride;

    MatrixView(const double* data_ptr,
               std::size_t row_count,
               std::size_t col_count,
               std::size_t row_stride_in,
               std::size_t col_stride_in)
        : data(data_ptr),
          rows(row_count),
          cols(col_count),
          row_stride(row_stride_in),
          col_stride(col_stride_in) {}

    MatrixView()
        : data(nullptr), rows(0), cols(0), row_stride(0), col_stride(0) {}
};

double dist_one_one(const double* p_data,
                    const double* q_data,
                    std::size_t length,
                    const DistConfig& config);

void dist_one_many(const double* reference,
                   std::size_t reference_length,
                   const MatrixView& others,
                   DistributionAxis axis,
                   double* out,
                   const DistConfig& config,
                   int num_threads);

void dist_many_many(const MatrixView& left,
                    DistributionAxis left_axis,
                    const MatrixView& right,
                    DistributionAxis right_axis,
                    double* out,
                    std::size_t out_leading_dim,
                    const DistConfig& config,
                    int num_threads);

void dist_self_symmetric(const MatrixView& data,
                         DistributionAxis axis,
                         double* out,
                         std::size_t out_leading_dim,
                         const DistConfig& config,
                         int num_threads);

void estimate_prob_empirical(const double* counts,
                             std::size_t length,
                             double* out);

// Note: resolve_num_threads and validate_p_parameter are now in common_utils.h

} // namespace philentropy

#endif // PHILENTROPY_DIST_INTERFACE_H
