#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "dist_interface.h"

namespace py = pybind11;
using philentropy::DistConfig;
using philentropy::DistributionAxis;
using philentropy::MatrixView;

namespace {

void estimate_probability_empirical(py::array_t<double> counts) {
    auto buf = counts.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("counts must be a 1D array");
    }
    double* data = static_cast<double*>(buf.ptr);
    std::size_t length = static_cast<std::size_t>(buf.shape[0]);
    philentropy::estimate_prob_empirical(data, length, data);
}

py::array_t<double> distance_one_one(
    py::array_t<double> p,
    py::array_t<double> q,
    const std::string& method,
    const std::string& unit = "log",
    double epsilon = 0.00001,
    py::object p_param = py::none(),
    py::object est_prob = py::none()
) {
    auto p_buf = p.request();
    auto q_buf = q.request();
    
    if (p_buf.ndim != 1 || q_buf.ndim != 1) {
        throw std::invalid_argument("P and Q must be 1D arrays");
    }
    if (p_buf.shape[0] != q_buf.shape[0]) {
        throw std::invalid_argument("P and Q must have the same length");
    }

    py::array_t<double> p_copy = p;
    py::array_t<double> q_copy = q;
    
    if (!est_prob.is_none()) {
        std::string est_method = py::cast<std::string>(est_prob);
        if (est_method != "empirical") {
            throw std::invalid_argument("Only 'empirical' probability estimation is supported");
        }
        p_copy = py::array_t<double>(p_buf.shape[0]);
        q_copy = py::array_t<double>(q_buf.shape[0]);
        std::copy_n(static_cast<double*>(p_buf.ptr), p_buf.shape[0], 
                    static_cast<double*>(p_copy.request().ptr));
        std::copy_n(static_cast<double*>(q_buf.ptr), q_buf.shape[0], 
                    static_cast<double*>(q_copy.request().ptr));
        estimate_probability_empirical(p_copy);
        estimate_probability_empirical(q_copy);
    }

    DistConfig config;
    config.method = method;
    config.unit = unit;
    config.epsilon = epsilon;
    config.has_p = false;
    config.p = 0.0;
    
    if (!p_param.is_none()) {
        config.p = py::cast<double>(p_param);
        config.has_p = true;
        philentropy::validate_p_parameter(method, config.p);
    }

    auto p_final = p_copy.request();
    auto q_final = q_copy.request();
    
    double result = philentropy::dist_one_one(
        static_cast<const double*>(p_final.ptr),
        static_cast<const double*>(q_final.ptr),
        static_cast<std::size_t>(p_final.shape[0]),
        config
    );

    py::array_t<double> out(1);
    *static_cast<double*>(out.request().ptr) = result;
    return out;
}

py::array_t<double> distance_one_many(
    py::array_t<double> reference,
    py::array_t<double> others,
    const std::string& method,
    const std::string& unit = "log",
    double epsilon = 0.00001,
    py::object p_param = py::none(),
    py::object est_prob = py::none(),
    int num_threads = 0
) {
    auto ref_buf = reference.request();
    auto others_buf = others.request();
    
    if (ref_buf.ndim != 1) {
        throw std::invalid_argument("reference must be a 1D array");
    }
    if (others_buf.ndim != 2) {
        throw std::invalid_argument("others must be a 2D array (distributions in rows)");
    }
    if (ref_buf.shape[0] != others_buf.shape[1]) {
        throw std::invalid_argument("reference length must match others column count");
    }

    py::array_t<double> ref_copy = reference;
    py::array_t<double> others_copy = others;
    
    if (!est_prob.is_none()) {
        std::string est_method = py::cast<std::string>(est_prob);
        if (est_method != "empirical") {
            throw std::invalid_argument("Only 'empirical' probability estimation is supported");
        }
        ref_copy = py::array_t<double>(ref_buf.shape[0]);
        std::copy_n(static_cast<double*>(ref_buf.ptr), ref_buf.shape[0], 
                    static_cast<double*>(ref_copy.request().ptr));
        estimate_probability_empirical(ref_copy);
        
        others_copy = py::array_t<double>({others_buf.shape[0], others_buf.shape[1]});
        auto others_copy_buf = others_copy.request();
        std::copy_n(static_cast<double*>(others_buf.ptr), 
                    others_buf.shape[0] * others_buf.shape[1],
                    static_cast<double*>(others_copy_buf.ptr));
        
        for (ssize_t i = 0; i < others_buf.shape[0]; ++i) {
            py::array_t<double> row = py::array_t<double>(others_buf.shape[1]);
            auto row_buf = row.request();
            for (ssize_t j = 0; j < others_buf.shape[1]; ++j) {
                static_cast<double*>(row_buf.ptr)[j] = 
                    static_cast<double*>(others_copy_buf.ptr)[i * others_buf.shape[1] + j];
            }
            estimate_probability_empirical(row);
            for (ssize_t j = 0; j < others_buf.shape[1]; ++j) {
                static_cast<double*>(others_copy_buf.ptr)[i * others_buf.shape[1] + j] = 
                    static_cast<double*>(row_buf.ptr)[j];
            }
        }
    }

    DistConfig config;
    config.method = method;
    config.unit = unit;
    config.epsilon = epsilon;
    config.has_p = false;
    config.p = 0.0;
    
    if (!p_param.is_none()) {
        config.p = py::cast<double>(p_param);
        config.has_p = true;
        philentropy::validate_p_parameter(method, config.p);
    }

    int threads = philentropy::resolve_num_threads(num_threads);

    auto ref_final = ref_copy.request();
    auto others_final = others_copy.request();
    
    py::array_t<double> out(others_final.shape[0]);
    auto out_buf = out.request();

    MatrixView others_view(
        static_cast<const double*>(others_final.ptr),
        static_cast<std::size_t>(others_final.shape[0]),
        static_cast<std::size_t>(others_final.shape[1]),
        static_cast<std::size_t>(others_final.shape[1]),
        1
    );

    philentropy::dist_one_many(
        static_cast<const double*>(ref_final.ptr),
        static_cast<std::size_t>(ref_final.shape[0]),
        others_view,
        DistributionAxis::Rows,
        static_cast<double*>(out_buf.ptr),
        config,
        threads
    );

    return out;
}

py::array_t<double> distance_matrix(
    py::array_t<double> x,
    const std::string& method,
    const std::string& unit = "log",
    double epsilon = 0.00001,
    py::object p_param = py::none(),
    py::object est_prob = py::none(),
    int num_threads = 0,
    bool mute_message = false
) {
    auto buf = x.request();
    
    if (buf.ndim != 2) {
        throw std::invalid_argument("x must be a 2D array (distributions in rows)");
    }

    std::vector<ssize_t> shape = {buf.shape[1], buf.shape[0]};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * buf.shape[1]};
    py::array_t<double> x_transposed(shape, strides);
    auto trans_buf = x_transposed.request();
    
    for (ssize_t i = 0; i < buf.shape[0]; ++i) {
        for (ssize_t j = 0; j < buf.shape[1]; ++j) {
            static_cast<double*>(trans_buf.ptr)[j + i * buf.shape[1]] = 
                static_cast<double*>(buf.ptr)[i * buf.shape[1] + j];
        }
    }

    if (!est_prob.is_none()) {
        std::string est_method = py::cast<std::string>(est_prob);
        if (est_method != "empirical") {
            throw std::invalid_argument("Only 'empirical' probability estimation is supported");
        }
        for (ssize_t i = 0; i < trans_buf.shape[1]; ++i) {
            py::array_t<double> col = py::array_t<double>(trans_buf.shape[0]);
            auto col_buf = col.request();
            for (ssize_t j = 0; j < trans_buf.shape[0]; ++j) {
                static_cast<double*>(col_buf.ptr)[j] = 
                    static_cast<double*>(trans_buf.ptr)[i * trans_buf.shape[0] + j];
            }
            estimate_probability_empirical(col);
            for (ssize_t j = 0; j < trans_buf.shape[0]; ++j) {
                static_cast<double*>(trans_buf.ptr)[i * trans_buf.shape[0] + j] = 
                    static_cast<double*>(col_buf.ptr)[j];
            }
        }
    }

    DistConfig config;
    config.method = method;
    config.unit = unit;
    config.epsilon = epsilon;
    config.has_p = false;
    config.p = 0.0;
    
    if (!p_param.is_none()) {
        config.p = py::cast<double>(p_param);
        config.has_p = true;
        philentropy::validate_p_parameter(method, config.p);
    }

    int threads = philentropy::resolve_num_threads(num_threads);

    if (!mute_message) {
        py::print("Metric: '" + method + "' with unit: '" + unit + 
                  "'; comparing: " + std::to_string(buf.shape[0]) + " vectors");
    }

    std::size_t n_dists = static_cast<std::size_t>(buf.shape[0]);
    py::array_t<double> out({static_cast<ssize_t>(n_dists), static_cast<ssize_t>(n_dists)});
    auto out_buf = out.request();

    MatrixView view(
        static_cast<const double*>(trans_buf.ptr),
        static_cast<std::size_t>(trans_buf.shape[0]),
        static_cast<std::size_t>(trans_buf.shape[1]),
        1,
        static_cast<std::size_t>(trans_buf.shape[0])
    );

    philentropy::dist_self_symmetric(
        view,
        DistributionAxis::Columns,
        static_cast<double*>(out_buf.ptr),
        n_dists,
        config,
        threads
    );

    return out;
}

std::vector<std::string> get_dist_methods() {
    return {
        "euclidean", "manhattan", "minkowski", "chebyshev",
        "sorensen", "gower", "soergel", "kulczynski_d",
        "canberra", "lorentzian", "intersection", "non-intersection",
        "wavehedges", "czekanowski", "motyka", "kulczynski_s",
        "tanimoto", "ruzicka", "inner_product", "harmonic_mean",
        "cosine", "hassebrook", "jaccard", "dice", "fidelity",
        "bhattacharyya", "hellinger", "matusita", "squared_chord",
        "squared_euclidean", "pearson", "neyman", "squared_chi",
        "prob_symm", "divergence", "clark", "additive_symm",
        "kullback-leibler", "jeffreys", "k_divergence", "topsoe",
        "jensen-shannon", "jensen_difference", "taneja",
        "kumar-johnson", "avg"
    };
}

} // namespace

PYBIND11_MODULE(philentropy_cpp, m) {
    m.doc() = "Python bindings for philentropy distance computation";

    m.def("distance_one_one", &distance_one_one,
          py::arg("p"),
          py::arg("q"),
          py::arg("method"),
          py::arg("unit") = "log",
          py::arg("epsilon") = 0.00001,
          py::arg("p") = py::none(),
          py::arg("est_prob") = py::none(),
          "Compute distance between two probability distributions");

    m.def("distance_one_many", &distance_one_many,
          py::arg("reference"),
          py::arg("others"),
          py::arg("method"),
          py::arg("unit") = "log",
          py::arg("epsilon") = 0.00001,
          py::arg("p") = py::none(),
          py::arg("est_prob") = py::none(),
          py::arg("num_threads") = 0,
          "Compute distances from one reference to many distributions");

    m.def("distance_matrix", &distance_matrix,
          py::arg("x"),
          py::arg("method"),
          py::arg("unit") = "log",
          py::arg("epsilon") = 0.00001,
          py::arg("p") = py::none(),
          py::arg("est_prob") = py::none(),
          py::arg("num_threads") = 0,
          py::arg("mute_message") = false,
          "Compute pairwise distance matrix for distributions");

    m.def("get_dist_methods", &get_dist_methods,
          "Get list of available distance methods");

    m.def("estimate_probability_empirical", &estimate_probability_empirical,
          py::arg("counts"),
          "Estimate probabilities from counts using empirical method (in-place)");
}
