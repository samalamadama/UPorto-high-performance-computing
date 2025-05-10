#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <numeric>
#include <omp.h>
#include <vector>

// t = 1
int main(int argc, char *argv[]) {
  double constexpr tight_b_t{1.};
  double constexpr final_time{100};
  int constexpr T_divisions{1024};
  double constexpr E{.1};
  int constexpr N{2048};
  double constexpr n_0{static_cast<double>(N) / 2};
  double constexpr k_0{1.};
  double constexpr stddev{static_cast<double>(N) / 32};
  Eigen::MatrixXd results(T_divisions, N);

  int number_of_threads = 8;
  if (argc > 1) {
    number_of_threads = std::stoi(argv[1]);
  }
  omp_set_num_threads(number_of_threads);

  //"y" values
  std::vector<double> t_range(T_divisions);
  std::generate(
      t_range.begin(), t_range.end(),
      [t = 0., dt = final_time / T_divisions]() mutable { return t += dt; });

  //"x" values
  std::vector<double> n_range(N);
  std::generate(n_range.begin(), n_range.end(),
                [n = 0., dn = 1]() mutable { return n += dn; });

  Eigen::VectorXcd psi_0_nbasis(N);
  std::transform(n_range.begin(), n_range.end(), psi_0_nbasis.begin(),
                 [n_0, k_0, stddev](int n) {
                   std::complex<double> value = std::exp(-std::complex(
                       std::pow((n - n_0) / stddev, 2) / 2, k_0 * n));
                   return value;
                 });
  // normalize
  {
    double normalization =
        std::accumulate(psi_0_nbasis.begin(), psi_0_nbasis.end(), 0.,
                        [](double sum, std::complex<double> coefficient) {
                          return sum + std::pow(std::abs(coefficient), 2);
                        });
    std::transform(psi_0_nbasis.begin(), psi_0_nbasis.end(),
                   psi_0_nbasis.begin(),
                   [normalization](std::complex<double> coefficient) {
                     return coefficient / normalization;
                   });
  }

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
  for (int row_index = 0; row_index != N; ++row_index) {
    for (int trig_index = -1; trig_index != 2; ++trig_index) {
      int col_index = row_index + trig_index;
      if (col_index != -1 && col_index != N) {
        H(row_index, col_index) = trig_index == 0 ? row_index * E : -tight_b_t;
      }
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(H);
  Eigen::VectorXcd psi_0_eigenbasis =
      eigen_solver.eigenvectors().transpose() * psi_0_nbasis;

#pragma omp parallel for schedule(dynamic)                                     \
    firstprivate(psi_0_eigenbasis, eigen_solver) shared(results)
  for (int t_index = 0; t_index != T_divisions; ++t_index) {
    Eigen::VectorXcd psi_t_eigenbasis(N);
    std::transform(
        psi_0_eigenbasis.begin(), psi_0_eigenbasis.end(),
        eigen_solver.eigenvalues().begin(), psi_t_eigenbasis.begin(),
        [t_index, t_range](std::complex<double> initial_coef, double exponent) {
          return initial_coef * std::exp(std::complex<double>(
                                    0., exponent * t_range[t_index]));
        });
    Eigen::VectorXcd psi_t_nbasis =
        eigen_solver.eigenvectors() * psi_t_eigenbasis;
    Eigen::VectorXd norm_psi_t_nbasis =
        psi_t_nbasis.unaryExpr([](std::complex<double> initial_coef) {
          return std::abs(initial_coef);
        });
    results.row(t_index) = norm_psi_t_nbasis.transpose();
  }

  write_matrix_to_CSV("1D_Efield_gaussian_evolution", results, n_range,
                      t_range);
}