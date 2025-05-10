#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"
#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

HPC::Tightly_bound_system create_1D_anderson(const std::array<int, 3> sheet_dim,
                                             const double W, const int seed) {
  constexpr std::complex<double> t(1, 0.);
  const HPC::Lattice generic_n_atoms_basis_lattice(sheet_dim[0], {1, 1, 1});
  std::uniform_real_distribution<double> distribution(-W / 2, W / 2);
  std::default_random_engine engine(seed);
  std::vector<HPC::Hopping> hoppings_anderson;

  for (int atom_index{0}; atom_index != sheet_dim[0] - 1; ++atom_index) {
    hoppings_anderson.push_back(
        {atom_index, atom_index, {0, 0, 0}, distribution(engine)});
    hoppings_anderson.push_back({atom_index, atom_index + 1, {0, 0, 0}, t});
  }

  hoppings_anderson.push_back(
      {sheet_dim[0] - 1, sheet_dim[0] - 1, {0, 0, 0}, distribution(engine)});
  hoppings_anderson.push_back({sheet_dim[0] - 1, 0, {1, 0, 0}, t});

  return HPC::Tightly_bound_system(generic_n_atoms_basis_lattice,
                                   hoppings_anderson);
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
  constexpr double stddev{0.01};
  std::array<int, 3> sheet_dimensions{std::stoi(argv[1]), 1, 1};
  int montecarlo_iterations{std::stoi(argv[2])};
  int random_average_iterations{std::stoi(argv[3])};
  constexpr int DOS_divisions{10000};
  constexpr double W{0.5};
  std::fstream seed_file("./exercise_8_data/seeds.txt", std::ios_base::in);
  std::vector<int> seeds(random_average_iterations);
  for (int i{0}; i != random_average_iterations; ++i) {
    seed_file >> seeds[i];
  }

  std::uniform_real_distribution<double> distribution(0., 1.);
  std::default_random_engine montecarlo_engine(203429412);

  std::vector<std::pair<double, double>> anderson_DOS(DOS_divisions);
  std::generate(anderson_DOS.begin(), anderson_DOS.end(),
                [min_eigenvalue = -3., range_step = 6. / DOS_divisions,
                 iteration_counter = 0]() mutable {
                  double current_range_value =
                      min_eigenvalue + range_step * iteration_counter;
                  ++iteration_counter;
                  return std::make_pair(current_range_value, 0.);
                });

  auto lorentzian = [stddev](double mode, double x) {
    return stddev / (M_PI * (std::pow(x - mode, 2) + std::pow(stddev, 2)));
  };

  for (const int &seed : seeds) {
    HPC::Tightly_bound_system anderson{
        create_1D_anderson(sheet_dimensions, W, seed)};
    HPC::Hamiltonian_builder anderson_builder(anderson);

    Eigen::VectorXcd cumulative_eigenvalues(montecarlo_iterations *
                                            sheet_dimensions[0]);

    int temp_index{0};
    for (int montecarlo_index{0}; montecarlo_index != montecarlo_iterations;
         ++montecarlo_index) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(
          anderson_builder.twistedbc_hamiltonian(
              distribution(montecarlo_engine), 0., 0.),
          Eigen::EigenvaluesOnly);
      auto eigenvalues = solver.eigenvalues();
      cumulative_eigenvalues.segment(temp_index, eigenvalues.size()) =
          eigenvalues;
      temp_index += eigenvalues.size();
    }
    add_DOS(anderson_DOS, cumulative_eigenvalues, lorentzian);
  }

  write_pairs_to_CSV("exercise_8_data/anderson_twisted_DOS", anderson_DOS);
}