#ifndef CSV_WRITER
#define CSV_WRITER

#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 *@brief Writes each element of the matrix to a csv file
 *output files have the shape:
 *0.0,x_1,x_2,x_3, (...) ,x_n
 *y_1,z_11,z_12,  (...)  ,z_1n
 *y_2, (...)    (...)    (...)
 *(...)
 *y_N,z_N,      (...)    ,z_Nn
 *
 *@param filename name of the file
 *@param matrixx matrix to be written to csv
 *@param x_range the range of values to which the columns of the matrix refer
 *@param y_range the range of values to which the rows of the matrix refer
 */

template <typename Derived, typename X_type, typename X_allocator,
          typename Y_type, typename Y_allocator>
void write_matrix_to_CSV(std::string filename,
                         Eigen::MatrixBase<Derived> const &matrix,
                         std::vector<X_type, X_allocator> const &x_range,
                         std::vector<Y_type, Y_allocator> const &y_range) {
  std::filesystem::path file_path(filename);
  std::filesystem::create_directories(file_path.parent_path());
  std::ofstream outfile(filename);

  int const N_rows = matrix.rows();
  int const N_cols = matrix.cols();
  if (static_cast<std::vector<int>::size_type>(N_rows) != y_range.size() ||
      static_cast<std::vector<int>::size_type>(N_cols) != x_range.size()) {
    throw std::runtime_error(
        "write_matrix_to_CVS matrix size does not match x or y range provided");
  }

  outfile << 0.0;
  for (const auto &x_value : x_range) {
    outfile << "," << x_value;
  }
  outfile << '\n';

  for (int row_index{0}; row_index < N_rows; ++row_index) {
    // y values
    outfile << y_range[row_index];
    for (int col_index{0}; col_index < N_cols; ++col_index) {
      // z values
      outfile << "," << matrix(row_index, col_index);
    }
    outfile << '\n';
  }
}

template <typename X_type, typename Y_type>
void write_pairs_to_CSV(std::string filename,
                        std::vector<std::pair<X_type, Y_type>> pairs) {
  std::filesystem::path file_path(filename);
  std::filesystem::create_directories(file_path.parent_path());

  std::ofstream outfile(filename);
  for (auto const &pair : pairs) {
    outfile << pair.first << ',' << pair.second << '\n';
  }
}

class Progress_bar {
  std::string _name;

public:
  Progress_bar() : _name{"Progress"} { update(0.); }

  Progress_bar(std::string name) : _name{name} { update(0.); }

  ~Progress_bar() { std::cout << std::endl; }

  void update(double progress) {
    std::cout << _name + " [";
    int barWidth = 70;
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        std::cout << "|";
      else
        std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
  }
};

#endif