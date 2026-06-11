#ifndef philentropy_Gower_Internal_H
#define philentropy_Gower_Internal_H

#include <cmath>
#include <algorithm>
#include <limits>

template<typename Row1, typename Row2>
inline void gower_numeric_pair_accumulate(
    const Row1& row1, const Row2& row2, int num_cols,
    const double* ranges,
    const double* weights,
    double& sum_dist,
    double& sum_weight
) {
    for (int i = 0; i < num_cols; ++i) {
        if (!std::isnan(row1[i]) && !std::isnan(row2[i])) {
            double w = weights ? weights[i] : 1.0;
            double r = ranges[i];
            if (r > 0) {
                sum_dist += w * std::abs(row1[i] - row2[i]) / r;
            }
            // Accumulate weight even if r==0 (zero-range feature: distance=0)
            sum_weight += w;
        }
    }
}

template<typename NumRow, typename CatRow>
inline double gower_mixed_row_pair(
    const NumRow& num1, const NumRow& num2, int num_num_cols,
    const double* ranges, const double* w_num,
    const CatRow& cat1, const CatRow& cat2, int num_cat_cols,
    const double* w_cat
) {
    double total_dist = 0.0;
    double total_weight = 0.0;

    // Numeric features via shared kernel
    gower_numeric_pair_accumulate(num1, num2, num_num_cols, ranges, w_num, total_dist, total_weight);

    // Categorical features (NA sentinel: R's NA_integer_ representation)
    const int NA_SENTINEL = std::numeric_limits<int>::min();
    for (int i = 0; i < num_cat_cols; ++i) {
        if (cat1[i] != NA_SENTINEL && cat2[i] != NA_SENTINEL) {
            double dist = (cat1[i] == cat2[i]) ? 0.0 : 1.0;
            total_dist += w_cat[i] * dist;
            total_weight += w_cat[i];
        }
    }

    if (total_weight == 0.0) {
        return 0.0;
    }

    return total_dist / total_weight;
}

#endif // philentropy_Gower_Internal_H
