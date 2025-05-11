#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/benchmarker.hpp"
#include <omp.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <fstream>
#include <string>
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


void compute_Hofstadter_butterfly() {
  const int lattice_size{200};
  constexpr int B_divisions{200};
  constexpr double k_x{2 * M_PI * 0.5};

  Eigen::MatrixXd Hofstadter_butterfly =
      Eigen::MatrixXd::Zero(lattice_size, B_divisions);
  std::vector<double> B_values(B_divisions);

#pragma omp parallel for firstprivate(lattice_size, B_divisions, k_x)          \
    shared(B_values, Hofstadter_butterfly)
  for (int B_index = 0; B_index < B_divisions; ++B_index) {
    double current_B = B_index * 2 * M_PI / B_divisions;
    B_values[B_index] = current_B;
    HPC::Tightly_bound_system peierls{
        create_1D_peierls(lattice_size, k_x, current_B)};

    HPC::Hamiltonian_builder peierls_builder(peierls);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> peierls_solver(
        peierls_builder.real_space_pbc_hamiltonian(), Eigen::EigenvaluesOnly);
    Hofstadter_butterfly.col(B_index) = peierls_solver.eigenvalues();
  }

  write_matrix_to_CSV("./exercise_9_data/Hofstadter_butterfly",
                      Hofstadter_butterfly, B_values,
                      std::vector<double>(lattice_size));
}

int main() {
    Progress_bar progress_bar("benchmarking progress");
    for (int thread_number{1}; thread_number != 9; ++thread_number) {
      omp_set_num_threads(thread_number);
      HPC::Benchmarker<std::string> benchmarker("Peierls begins");
      compute_Hofstadter_butterfly();
      benchmarker.save_time("Peierls ends");
      std::ofstream benchmark_file("exercise_9_data/Peierls_benchmark_" +
                                   std::to_string(thread_number) + "_threads");
      benchmarker.print_time_intervals<std::chrono::nanoseconds>(benchmark_file);
      progress_bar.update(static_cast<double>(thread_number) / 8);
    }
}