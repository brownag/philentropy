#include "dist_interface.h"
#include "common_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "dist_dispatch.h"

namespace philentropy {
namespace {

int default_thread_count() {
    const char* env = std::getenv("RCPP_PARALLEL_NUM_THREADS");
    if (env != nullptr) {
        int parsed = std::atoi(env);
        if (parsed > 0) {
            return parsed;
        }
    }
    return 2;
}

class ThreadRangeExecutor {
  public:
    ThreadRangeExecutor(std::size_t begin,
                        std::size_t end,
                        int threads) {
        if (end < begin) {
            end = begin;
        }
        begin_ = begin;
        end_ = end;
        threads_ = threads <= 0 ? 1 : threads;
        if (threads_ > static_cast<int>(end_ - begin_)) {
            threads_ = static_cast<int>(end_ - begin_);
            if (threads_ <= 0) {
                threads_ = 1;
            }
        }
    }

    template <typename Func>
    void run(Func&& func) const {
        if (threads_ <= 1 || (end_ - begin_) <= 1) {
            func(begin_, end_);
            return;
        }

        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads_));
        std::size_t total = end_ - begin_;
        std::size_t block = total / static_cast<std::size_t>(threads_);
        std::size_t remainder = total % static_cast<std::size_t>(threads_);
        std::size_t current = begin_;
        for (int i = 0; i < threads_; ++i) {
            std::size_t chunk = block + (i < static_cast<int>(remainder) ? 1 : 0);
            std::size_t local_begin = current;
            std::size_t local_end = current + chunk;
            current = local_end;
            pool.emplace_back([local_begin, local_end, &func]() {
                func(local_begin, local_end);
            });
        }
        for (auto& worker : pool) {
            worker.join();
        }
    }

  private:
    std::size_t begin_{};
    std::size_t end_{};
    int threads_{};
};

class StridedIterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = double;
    using difference_type = std::ptrdiff_t;
    using pointer = const double*;
    using reference = const double&;

    StridedIterator(const double* ptr, std::ptrdiff_t stride)
        : ptr_(ptr), stride_(stride) {}

    reference operator*() const { return *ptr_; }
    pointer operator->() const { return ptr_; }

    StridedIterator& operator++() {
        ptr_ += stride_;
        return *this;
    }

    StridedIterator operator++(int) {
        StridedIterator tmp(*this);
        ptr_ += stride_;
        return tmp;
    }

    StridedIterator& operator+=(difference_type n) {
        ptr_ += stride_ * n;
        return *this;
    }

    StridedIterator operator+(difference_type n) const {
        return StridedIterator(ptr_ + stride_ * n, stride_);
    }

    reference operator[](difference_type n) const {
        return *(ptr_ + stride_ * n);
    }

    difference_type operator-(const StridedIterator& other) const {
        return (ptr_ - other.ptr_) / stride_;
    }

    bool operator==(const StridedIterator& other) const {
        return ptr_ == other.ptr_;
    }

    bool operator!=(const StridedIterator& other) const {
        return ptr_ != other.ptr_;
    }

    bool operator<(const StridedIterator& other) const {
        return ptr_ < other.ptr_;
    }

  private:
    const double* ptr_;
    std::ptrdiff_t stride_;
};

struct DistributionView {
    const double* base;
    std::size_t length;
    std::ptrdiff_t stride;
};

DistributionView get_distribution(const MatrixView& matrix,
                                   DistributionAxis axis,
                                   std::size_t index) {
    if (axis == DistributionAxis::Columns) {
        return {matrix.data + index * matrix.col_stride,
                matrix.rows,
                static_cast<std::ptrdiff_t>(matrix.row_stride)};
    }
    return {matrix.data + index * matrix.row_stride,
            matrix.cols,
            static_cast<std::ptrdiff_t>(matrix.col_stride)};
}

double compute_distance(const double* reference,
                        std::size_t ref_length,
                        const double* reference_stride_ptr,
                        std::ptrdiff_t ref_stride,
                        const DistributionView& target,
                        const DistConfig& config) {
    StridedIterator ref_begin(reference_stride_ptr, ref_stride);
    StridedIterator ref_end = ref_begin + static_cast<std::ptrdiff_t>(ref_length);
    StridedIterator target_begin(target.base, target.stride);
    double p_value = config.has_p ? config.p : std::numeric_limits<double>::quiet_NaN();
    return dispatch_dist_internal(ref_begin,
                                  ref_end,
                                  target_begin,
                                  config.method,
                                  config.unit,
                                  config.epsilon,
                                  p_value);
}

double compute_distance(const DistributionView& lhs,
                        const DistributionView& rhs,
                        const DistConfig& config) {
    StridedIterator lhs_begin(lhs.base, lhs.stride);
    StridedIterator lhs_end = lhs_begin + static_cast<std::ptrdiff_t>(lhs.length);
    StridedIterator rhs_begin(rhs.base, rhs.stride);
    double p_value = config.has_p ? config.p : std::numeric_limits<double>::quiet_NaN();
    return dispatch_dist_internal(lhs_begin,
                                  lhs_end,
                                  rhs_begin,
                                  config.method,
                                  config.unit,
                                  config.epsilon,
                                  p_value);
}

} // namespace

// Note: resolve_num_threads and validate_p_parameter moved to common_utils.h

double dist_one_one(const double* p_data,
                    const double* q_data,
                    std::size_t length,
                    const DistConfig& config) {
    StridedIterator p_begin(p_data, 1);
    StridedIterator p_end = p_begin + static_cast<std::ptrdiff_t>(length);
    StridedIterator q_begin(q_data, 1);
    double p_value = config.has_p ? config.p : std::numeric_limits<double>::quiet_NaN();
    return dispatch_dist_internal(p_begin,
                                  p_end,
                                  q_begin,
                                  config.method,
                                  config.unit,
                                  config.epsilon,
                                  p_value);
}

void dist_one_many(const double* reference,
                   std::size_t reference_length,
                   const MatrixView& others,
                   DistributionAxis axis,
                   double* out,
                   const DistConfig& config,
                   int num_threads) {
    ThreadRangeExecutor executor(0, axis == DistributionAxis::Columns ? others.cols : others.rows,
                                 resolve_num_threads(num_threads));
    executor.run([&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            DistributionView target = get_distribution(others, axis, idx);
            if (target.length != reference_length) {
                throw std::invalid_argument("Input vectors do not have the same length.");
            }
            out[idx] = compute_distance(reference,
                                        reference_length,
                                        reference,
                                        1,
                                        target,
                                        config);
        }
    });
}

void dist_many_many(const MatrixView& left,
                    DistributionAxis left_axis,
                    const MatrixView& right,
                    DistributionAxis right_axis,
                    double* out,
                    std::size_t out_leading_dim,
                    const DistConfig& config,
                    int num_threads) {
    std::size_t left_count = (left_axis == DistributionAxis::Columns) ? left.cols : left.rows;
    std::size_t right_count = (right_axis == DistributionAxis::Columns) ? right.cols : right.rows;
    ThreadRangeExecutor executor(0, left_count, resolve_num_threads(num_threads));
    executor.run([&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            DistributionView lhs = get_distribution(left, left_axis, i);
            for (std::size_t j = 0; j < right_count; ++j) {
                DistributionView rhs = get_distribution(right, right_axis, j);
                if (lhs.length != rhs.length) {
                    throw std::invalid_argument("Input vectors do not have the same length.");
                }
                double value = compute_distance(lhs, rhs, config);
                out[j * out_leading_dim + i] = value;
            }
        }
    });
}

void dist_self_symmetric(const MatrixView& data,
                         DistributionAxis axis,
                         double* out,
                         std::size_t out_leading_dim,
                         const DistConfig& config,
                         int num_threads) {
    std::size_t count = (axis == DistributionAxis::Columns) ? data.cols : data.rows;
    ThreadRangeExecutor executor(0, count, resolve_num_threads(num_threads));
    executor.run([&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            DistributionView lhs = get_distribution(data, axis, i);
            for (std::size_t j = i; j < count; ++j) {
                DistributionView rhs = get_distribution(data, axis, j);
                if (lhs.length != rhs.length) {
                    throw std::invalid_argument("Input vectors do not have the same length.");
                }
                double value = compute_distance(lhs, rhs, config);
                out[j * out_leading_dim + i] = value;
                out[i * out_leading_dim + j] = value;
            }
        }
    });
}

void estimate_prob_empirical(const double* counts,
                             std::size_t length,
                             double* out) {
    double sum = 0.0;
    for (std::size_t i = 0; i < length; ++i) {
        sum += counts[i];
    }
    if (sum == 0.0) {
        throw std::invalid_argument("Count vector sums to zero; cannot estimate probabilities.");
    }
    for (std::size_t i = 0; i < length; ++i) {
        out[i] = counts[i] / sum;
    }
}

} // namespace philentropy
