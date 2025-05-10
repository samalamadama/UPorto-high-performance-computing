#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <functional>
#include <numeric>
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

  for (auto &current_point : evaluated_dos) {
    for (const double &energy_mean : real_eigenvalues) {
      current_point.second +=
          delta_approximation(energy_mean, current_point.first);
    }
  }
  return evaluated_dos;
}

int main() {
  constexpr double stddev{0.1};
  constexpr std::array<int, 3> sheet_dimensions{16, 16, 1};
  constexpr std::array<int, 3> supercell_sheet_dimensions{16, 16 , 1};

  const int cells_per_sheet =
      std::accumulate(sheet_dimensions.begin(), sheet_dimensions.end(), 1,
                      std::multiplies<int>());
  const int supercells_number = std::accumulate(
      supercell_sheet_dimensions.begin(), supercell_sheet_dimensions.end(), 1,
      std::multiplies<int>());

  HPC::Tightly_bound_system BN{create_BN_system(sheet_dimensions)};
  HPC::Hamiltonian_builder BN_builder(BN);

  Eigen::VectorXcd cumulative_eigenvalues(2*cells_per_sheet * supercells_number);

  int temp_index{0};
  for (int x{0}; x != supercell_sheet_dimensions[0]; ++x) {
    for (int y{0}; y != supercell_sheet_dimensions[1]; ++y) {
      auto H = BN_builder.twistedbc_hamiltonian(
          2 * M_PI * x / supercell_sheet_dimensions[0],
          2 * M_PI * y / supercell_sheet_dimensions[1],
          2 * M_PI * 0 / supercell_sheet_dimensions[2]);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen_solver(
          H, Eigen::EigenvaluesOnly);
      auto eigenvalues= eigen_solver.eigenvalues();
      cumulative_eigenvalues.segment(temp_index, eigenvalues.size())=eigenvalues;
      temp_index+=eigenvalues.size();
    }
  }

  auto lorentzian = [stddev](double mode, double x) {
    return stddev / (M_PI * (std::pow(x - mode, 2) + std::pow(stddev, 2)));
  };

  auto BN_DOS = get_DOS(cumulative_eigenvalues, 1000, 0.5, lorentzian);
  write_pairs_to_CSV("exercise_7_data/BN_twisted_DOS", BN_DOS);
}