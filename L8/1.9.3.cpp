#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

HPC::Tightly_bound_system create_1D_peierls(const int lattice_size,
                                            const double k_x, const double B) {
  constexpr std::complex<double> t(1, 0.);
  const HPC::Lattice n_atoms_basis_lattice_1D(lattice_size, {1, 1, 1});

  std::vector<HPC::Hopping> hoppings_peierls;
  hoppings_peierls.reserve(2 * lattice_size);

  for (int atom_index{0}; atom_index != lattice_size - 1; ++atom_index) {
    hoppings_peierls.push_back({atom_index,
                                atom_index,
                                {0, 0, 0},
                                2 * std::cos(k_x - B * atom_index)});
    hoppings_peierls.push_back({atom_index, atom_index + 1, {0, 0, 0}, -t});
  }
  hoppings_peierls.push_back({lattice_size - 1,
                              lattice_size - 1,
                              {0, 0, 0},
                              2 * std::cos(k_x - B * lattice_size)});
  hoppings_peierls.push_back({lattice_size - 1, 0, {0, 0, 0}, -t});

  return HPC::Tightly_bound_system(n_atoms_basis_lattice_1D, hoppings_peierls);
}

void add_DOS(
    std::vector<std::pair<double, double>> &storage_DOS,
    const Eigen::VectorXcd &eigenvalues,
    std::function<double(double, double)> delta_approximation = [](double mode,
                                                                   double x) {
      return 0.1 / (M_PI * (std::pow(x - mode, 2) + std::pow(0.1, 2)));
    }) {
  Eigen::VectorXd real_eigenvalues = eigenvalues.real();

  for (auto &current_point : storage_DOS) {
    for (const double &energy_mean : real_eigenvalues) {
      current_point.second +=
          delta_approximation(energy_mean, current_point.first);
    }
  }
}

int main(int argc, char *argv[]) {
  const int lattice_size{1000};
  constexpr int B_divisions{1000};
  constexpr double k_x{2 * M_PI * 0.5};

  Eigen::MatrixXd Hofstadter_butterfly =
      Eigen::MatrixXd::Zero(lattice_size, B_divisions);
  std::vector<double> B_values(B_divisions);

  Progress_bar progress_bar("Peierls");
#pragma omp parallel for firstprivate(lattice_size, B_divisions, k_x)          \
    shared(B_values, Hofstadter_butterfly)
  for (int B_index; B_index < B_divisions; ++B_index) {
    double current_B = B_index * 2 * M_PI / B_divisions;
    B_values[B_index] = current_B;
    HPC::Tightly_bound_system peierls{
        create_1D_peierls(lattice_size, k_x, current_B)};

    HPC::Hamiltonian_builder peierls_builder(peierls);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> peierls_solver(
        peierls_builder.real_space_pbc_hamiltonian(), Eigen::EigenvaluesOnly);
    Hofstadter_butterfly.col(B_index) = peierls_solver.eigenvalues();
    progress_bar.update(static_cast<double>(B_index) / B_divisions);
  }

  write_matrix_to_CSV("./exercise_9_data/Hofstadter_butterfly",
                      Hofstadter_butterfly, B_values,
                      std::vector<double>(lattice_size));
}