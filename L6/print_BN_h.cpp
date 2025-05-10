#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/hamiltonian_builder.hpp"
#include <iostream>


HPC::Tightly_bound_system create_BN_system(const std::array<int, 3> sheet_dim) {
    enum Atom_type { A, B };
    constexpr double delta{0.};
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

  int main(){
    constexpr std::array<int, 3> sheet_dimensions{12, 12, 1};

  HPC::Tightly_bound_system BN{create_BN_system(sheet_dimensions)};
  HPC::Hamiltonian_builder BN_builder(BN);
  auto H = BN_builder.real_space_pbc_hamiltonian();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen_solver(H, Eigen::EigenvaluesOnly);
  }