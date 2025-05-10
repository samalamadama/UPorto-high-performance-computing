#ifndef HAMILTONIAN_BUILDER
#define HAMILTONIAN_BUILDER

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace HPC {
using Position = std::array<double, 3>;
using Lattice_vector = std::array<int, 3>;

Lattice_vector inline operator+(const Lattice_vector &a,
                                const Lattice_vector &b) {
  Lattice_vector result;
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>());
  return result;
}

struct Atom {
  int info; // info type to be decided

  Atom(int info_) : info(info_) {}
  Atom() : info(0) {}
};

struct Lattice {
private:
  using Basis_element = std::pair<Atom, Position>;

public:
  const std::array<Position, 3> primitive_vectors;
  const std::vector<Basis_element> basis;
  const std::array<int, 3> atoms_per_dimension;

  Lattice()
      : Lattice({Position{1., 0, 0}, {0., 1., 0.}, {0., 0., 1.}},
                {Basis_element()}, {100, 100, 100}) {};

  Lattice(std::array<Position, 3> primitives, std::array<int, 3> dimensions)
      : Lattice(primitives, {Basis_element()}, dimensions) {};

  Lattice(std::array<Position, 3> primitives, int number_of_basis_atoms,
          std::array<int, 3> dimensions)
      : Lattice(primitives, std::vector<Basis_element>(number_of_basis_atoms),
                dimensions) {};

  Lattice(int number_of_basis_atoms, std::array<int, 3> dimensions)
      : Lattice({Position{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}},
                std::vector<Basis_element>(number_of_basis_atoms),
                dimensions) {};

  Lattice(std::array<Position, 3> primitives,
          std::vector<std::pair<Atom, Position>> basis_,
          std::array<int, 3> dimensions)
      : primitive_vectors(primitives), basis(basis_),
        atoms_per_dimension(dimensions) {
    if (std::any_of(dimensions.begin(), dimensions.end(),
                    [](int i) { return i <= 0; }))
      throw std::runtime_error(
          "number of lattice cells given is zero or negative");
    if (basis_.empty())
      throw std::runtime_error("basis must contain one atom");
  };
};

using Atom_index = int;

struct Hopping {
  Atom_index reference_atom_index;
  Atom_index hopping_atom_index;
  Lattice_vector cell_offset;
  std::complex<double> amplitude;

  Hopping(Atom_index reference_atom, Atom_index hopping_atom,
          Lattice_vector position_, std::complex<double> amplitude_)
      : reference_atom_index(reference_atom), hopping_atom_index(hopping_atom),
        cell_offset(position_), amplitude(amplitude_) {
    if (reference_atom < 0 || hopping_atom < 0) {
      throw std::runtime_error("negative indexes for atoms are not allowed");
    }
  }
};

struct Tightly_bound_system {
  Lattice lattice;
  std::vector<Hopping> hoppings_list;

  Tightly_bound_system(const Lattice &lattice_,
                       const std::vector<Hopping> &hoppings)
      : lattice(lattice_), hoppings_list(hoppings) {
    int basis_size = (int)lattice.basis.size();
    for (const Hopping &hopping : hoppings) {
      if (hopping.reference_atom_index >= basis_size ||
          hopping.hopping_atom_index >= basis_size)
        throw std::runtime_error(
            "hoppings to be introduced contain invalid indeces");
    }
  }

  void add_hopping(Atom_index reference_atom, Atom_index hopping_atom,
                   Lattice_vector position_, std::complex<double> amplitude_) {
    int basis_size = (int)lattice.basis.size();
    if (reference_atom >= basis_size || hopping_atom >= basis_size)
      throw std::runtime_error(
          "hopping to be added contains invalid atom indeces");
    Hopping hopping(reference_atom, hopping_atom, position_, amplitude_);
    hoppings_list.push_back(hopping);
  }
};

class Hamiltonian_builder {
  const Tightly_bound_system _system;
  Eigen::MatrixXcd _base_hamiltonian;

  std::array<bool, 3> check_in_range(Lattice_vector position) const {
    std::array<bool, 3> is_in_range;
    for (int direction{0}; direction != 3; ++direction) {
      is_in_range[direction] =
          position[direction] <
              _system.lattice.atoms_per_dimension[direction] &&
          position[direction] >= 0;
    }
    return is_in_range;
  }

  int pbc_index_calculator(Lattice_vector position,
                           Atom_index basis_index) const {
    int flattening_index{0};
    {
      int unwrapper{1};
      for (int index = 2; index != -1; --index) {
        int periodic_position =
            (position[index] + _system.lattice.atoms_per_dimension[index]) %
            _system.lattice.atoms_per_dimension[index];
        flattening_index += periodic_position * unwrapper;
        unwrapper *= _system.lattice.atoms_per_dimension[index];
      }
    }
    return flattening_index * (int)_system.lattice.basis.size() +
           (int)basis_index;
  }

  template <typename AmplitudeFunc>
  void set_out_of_bounds_elements(AmplitudeFunc calculate_amplitude) {
    for (int x{0}; x != _system.lattice.atoms_per_dimension[0]; ++x) {
      for (int y{0}; y != _system.lattice.atoms_per_dimension[1]; ++y) {
        for (int z{0}; z != _system.lattice.atoms_per_dimension[2]; ++z) {
          for (const Hopping &hopping : _system.hoppings_list) {
            Lattice_vector current_cell_vector = {x, y, z};
            Lattice_vector hopping_cell_vector{current_cell_vector +
                                               hopping.cell_offset};
            std::array<bool, 3> hopping_cell_is_in_range =
                check_in_range(hopping_cell_vector);
            if (!std::all_of(hopping_cell_is_in_range.begin(),
                             hopping_cell_is_in_range.end(),
                             [](bool direction_is_in_range) {
                               return direction_is_in_range;
                             })) {
              int current_cell_index = pbc_index_calculator(
                  current_cell_vector, hopping.reference_atom_index);
              int hopping_cell_index = pbc_index_calculator(
                  hopping_cell_vector, hopping.hopping_atom_index);
              _base_hamiltonian(current_cell_index, hopping_cell_index) =
                  calculate_amplitude(hopping, hopping_cell_is_in_range);
              _base_hamiltonian(hopping_cell_index, current_cell_index) =
                  std::conj(
                      calculate_amplitude(hopping, hopping_cell_is_in_range));
            }
          }
        }
      }
    }
  }

public:
  Hamiltonian_builder(const Tightly_bound_system &system) : _system{system} {
    int number_of_atoms = std::accumulate(
        _system.lattice.atoms_per_dimension.begin(),
        _system.lattice.atoms_per_dimension.end(), 1, std::multiplies<int>());
    int system_dimension{number_of_atoms * (int)_system.lattice.basis.size()};
    Eigen::MatrixXcd h =
        Eigen::MatrixXcd::Zero(system_dimension, system_dimension);

    for (int x{0}; x != _system.lattice.atoms_per_dimension[0]; ++x) {
      for (int y{0}; y != _system.lattice.atoms_per_dimension[1]; ++y) {
        for (int z{0}; z != _system.lattice.atoms_per_dimension[2]; ++z) {
          for (const Hopping &hopping : _system.hoppings_list) {
            Lattice_vector current_cell_vector = {x, y, z};
            Lattice_vector hopping_cell_vector{current_cell_vector +
                                               hopping.cell_offset};
            std::array<bool, 3> hopping_cell_is_in_range =
                check_in_range(hopping_cell_vector);
            if (std::all_of(hopping_cell_is_in_range.begin(),
                            hopping_cell_is_in_range.end(),
                            [](bool direction_is_in_range) {
                              return direction_is_in_range;
                            })) {
              int hopping_cell_index = pbc_index_calculator(
                  hopping_cell_vector, hopping.hopping_atom_index);
              int current_cell_index = pbc_index_calculator(
                  current_cell_vector, hopping.reference_atom_index);
              h(current_cell_index, hopping_cell_index) = hopping.amplitude;
              h(hopping_cell_index, current_cell_index) =
                  std::conj(hopping.amplitude);
            }
          }
        }
      }
    }
    _base_hamiltonian = h;
  }

  Eigen::MatrixXcd real_space_pbc_hamiltonian() {
    set_out_of_bounds_elements(
        [](Hopping hopping, [[maybe_unused]] std::array<bool, 3>) {
          return hopping.amplitude;
        });
    return _base_hamiltonian;
  }

  Eigen::MatrixXcd real_space_obc_hamiltonian() {
    set_out_of_bounds_elements(
        []([[maybe_unused]] Hopping, [[maybe_unused]] std::array<bool, 3>) {
          return std::complex<double>(0., 0.);
        });
    return _base_hamiltonian;
  }

  Eigen::MatrixXcd twistedbc_hamiltonian(const double k_x, const double k_y,
                                         const double k_z) {
    set_out_of_bounds_elements(
        [k_x, k_y, k_z](Hopping hopping,
                        std::array<bool, 3> hopping_cell_is_in_range) {
          std::complex<double> total_phase = std::exp(std::complex<double>(
              0., k_x * (int)!hopping_cell_is_in_range[0] +
                      k_y * (int)!hopping_cell_is_in_range[1] +
                      k_z * (int)!hopping_cell_is_in_range[2]));
          return hopping.amplitude * total_phase;
        });
    return _base_hamiltonian;
  };
};

} // namespace HPC

#endif