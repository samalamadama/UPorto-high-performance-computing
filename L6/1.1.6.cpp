#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/benchmarker.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <fstream>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

HPC::Tightly_bound_system create_BN_system(const std::array<int, 3> sheet_dim) {
  enum Atom_type { A, B };
  constexpr double delta{1.};
  constexpr std::complex<double> t(1, 0.);
  constexpr std::complex<double> E_A{delta / 2};
  constexpr std::complex<double> E_B{-delta / 2};

  const HPC::Lattice generic_2_atoms_basis_lattice(2, sheet_dim);
  const std::vector<HPC::Hopping> graphene_hoppings{{A, A, {0, 0, 0}, E_A},
                                                    {B, B, {0, 0, 0}, E_B},
                                                    {A, B, {0, 0, 0}, -t},
                                                    {A, B, {-1, 0, 0}, -t},
                                                    {A, B, {0, -1, 0}, -t}};
  return HPC::Tightly_bound_system(generic_2_atoms_basis_lattice,
                                   graphene_hoppings);
}

std::vector<std::pair<double, double>> get_DOS(
    const Eigen::VectorXcd &eigenvalues, const int range_divisions,
    double extra_range = 0.,
    std::function<double(double, double)> delta_approximation = [](double mode,
                                                                   double x) {
      return 0.1 / (M_PI * (std::pow(x - mode, 2) + std::pow(0.1, 2)));
    }) {
  Eigen::VectorXd real_eigenvalues = eigenvalues.real();
  double min_eigenvalue = real_eigenvalues.minCoeff();
  double range_step =
      (real_eigenvalues.maxCoeff() - min_eigenvalue + 2 * extra_range) /
      range_divisions;
  std::vector<std::pair<double, double>> evaluated_dos(range_divisions);
  std::generate(evaluated_dos.begin(), evaluated_dos.end(),
                [min_eigenvalue, extra_range, range_step,
                 iteration_counter = 0]() mutable {
                  double current_value = min_eigenvalue - extra_range +
                                         range_step * iteration_counter;
                  ++iteration_counter;
                  return std::make_pair(current_value, 0.);
                });

#pragma omp parallel for schedule(static, 4)                                   \
    firstprivate(real_eigenvalues, delta_approximation)
  for (auto &current_point : evaluated_dos) {
    for (const double &energy_mean : real_eigenvalues) {
      current_point.second +=
          delta_approximation(energy_mean, current_point.first);
    }
  }
  return evaluated_dos;
}

void compute_BN_DOS(HPC::Benchmarker<std::string> &benchmarker) {
  benchmarker.save_time("initialization begins");
  constexpr double stddev{0.01};
  constexpr std::array<int, 3> sheet_dimensions{24, 24, 1};

  HPC::Tightly_bound_system BN{create_BN_system(sheet_dimensions)};
  HPC::Hamiltonian_builder BN_builder(BN);
  auto H = BN_builder.real_space_pbc_hamiltonian();
  benchmarker.save_time("initialization ends");

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen_solver(H);
  benchmarker.save_time("matrix diagonalization");

  auto lorentzian = [stddev](double mode, double x) {
    return stddev / (M_PI * (std::pow(x - mode, 2) + std::pow(stddev, 2)));
  };

  auto BN_DOS = get_DOS(eigen_solver.eigenvalues(), 10000, 0.5, lorentzian);
  benchmarker.save_time("DOS computation");

  write_pairs_to_CSV("exercise_6_data/BN_DOS", BN_DOS);
  benchmarker.save_time("write to CSV");
}

int main() {
  Progress_bar progress_bar("BN DOS progress");
  for (int thread_number{1}; thread_number != 9; ++thread_number) {
    omp_set_num_threads(thread_number);
    HPC::Benchmarker<std::string> benchmarker("BN program begins");
    compute_BN_DOS(benchmarker);
    std::ofstream benchmark_file("exercise_6_data/BN_benchmark_improved_" +
                                 std::to_string(thread_number) + "_threads");
    benchmarker.print_time_intervals<std::chrono::nanoseconds>(benchmark_file);
    progress_bar.update(static_cast<double>(thread_number) / 8);
  }
}
