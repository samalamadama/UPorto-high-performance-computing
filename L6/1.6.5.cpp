#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/benchmarker.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <fstream>
#include <omp.h>
#include <string>
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

void compute_BN_local_DOS(HPC::Benchmarker<std::string> &benchmarker) {
  benchmarker.save_time("initialization begins");
  constexpr double stddev{0.01};
  constexpr int energy_divisions{15};
  constexpr std::array<int, 3> sheet_dimensions{24, 12, 1};
  constexpr int system_size =
      sheet_dimensions[0] * sheet_dimensions[1] * sheet_dimensions[2] * 2;

  HPC::Tightly_bound_system BN{create_BN_system(sheet_dimensions)};
  HPC::Hamiltonian_builder BN_builder(BN);
  auto H = BN_builder.real_space_pbc_hamiltonian();
  benchmarker.save_time("initialization ends");

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen_solver(H);
  auto eigenvalues = eigen_solver.eigenvalues();
  auto eigenvectors = eigen_solver.eigenvectors();

  std::vector<double> E_range(energy_divisions);
  std::generate(E_range.begin(), E_range.end(),
                [x = eigenvalues.minCoeff(),
                 dx = (eigenvalues.maxCoeff() - eigenvalues.minCoeff()) /
                      energy_divisions]() mutable { return x += dx; });

  std::vector<double> position_range(system_size);
  std::generate(position_range.begin(), position_range.end(),
                [x = 0, dx = 1]() mutable { return x += dx; });

  benchmarker.save_time("matrix diagonalization");

  auto lorentzian = [stddev](double mode, double x) {
    return stddev / (M_PI * (std::pow(x - mode, 2) + std::pow(stddev, 2)));
  };

  benchmarker.save_time("DOS computation");
  Eigen::VectorXd real_eigenvalues = eigen_solver.eigenvalues().real();
  Eigen::MatrixXd evaluated_local_dos =
      Eigen::MatrixXd::Zero(energy_divisions, system_size);

  for (int energy_index{0}; energy_index != energy_divisions; energy_index++) {
    Eigen::VectorXd single_energy_dos(system_size);
    for (int position_index{0}; position_index != system_size;
         position_index++) {
      for (int eigen_index{0}; eigen_index != system_size; ++eigen_index) {
        single_energy_dos(position_index) +=
            lorentzian(eigenvalues(eigen_index), E_range[energy_index]) *
            std::abs(eigenvectors(position_index, eigen_index));
      }
    }
    evaluated_local_dos.row(energy_index) = single_energy_dos.transpose();
  }
  write_matrix_to_CSV("./exercise_6_data/BN_Local_DOS", evaluated_local_dos,
                      position_range, E_range);
}

int main() {
  /*
Progress_bar progress_bar("BN local DOS progress");
int maximum_thread_number = 8;
for (int thread_number{1}; thread_number != maximum_thread_number + 1;
     ++thread_number) {
  omp_set_num_threads(thread_number);
  HPC::Benchmarker<std::string> benchmarker("BN program begins");
  compute_BN_local_DOS(benchmarker);
  std::ofstream benchmark_file("exercise_6_data/BN_local_benchmark" +
                               std::to_string(thread_number) + "_threads");
  benchmarker.print_time_intervals<std::chrono::nanoseconds>(benchmark_file);
  progress_bar.update(static_cast<double>(thread_number) / 8);
}
  */
  HPC::Benchmarker<std::string> benchmarker("BN program begins");
  std::ofstream benchmark_file("exercise_6_data/BN_local_benchmark");
  compute_BN_local_DOS(benchmarker);
  benchmarker.print_time_intervals<std::chrono::nanoseconds>(benchmark_file);
}
