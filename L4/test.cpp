
#include <cstdlib>
#include <cstring>
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>

class Lorentzian{
    private:
    double const _mode;
    double const _sigma;

    public:
    Lorentzian(double mode, double sigma) : _mode(mode), _sigma(sigma){}
    double operator()(double x) const {
        return _sigma/(M_PI*(std::pow(x-_mode,2)+std::pow(_sigma, 2)));
    }
};

template <typename Derived, typename X_type, typename X_allocator, typename Y_type, typename Y_allocator>
void write_matrix_to_CSV(std::string filename, Eigen::MatrixBase<Derived> const&matrix, 
                        std::vector<X_type, X_allocator> const&x_range, 
                        std::vector<Y_type, Y_allocator> const&y_range){
    std::ofstream outfile(filename);

    int N_rows = matrix.rows();
    int N_cols = matrix.cols();
    if(static_cast<std::vector<int>::size_type>(N_rows)!=y_range.size()||static_cast<std::vector<int>::size_type>(N_cols)!=x_range.size()){
        throw std::runtime_error("write_matrix_to_CVS matrix size does not match x or y range provided");
    }

    //print output files, it's in the form 
    //  0.0 x_1     x_2     x_3     ...     x_n
    //  y_1 z_11    z_12    ...             z_1n
    //  y_2 ...             ...             ...
    //  ... 
    //  y_N z_N1    ...                     z_Nn

    //x line of the output
    outfile<< 0.0;
    for(int col_index=0; col_index<N_cols; ++col_index){
        outfile<<","<<x_range[col_index];
    }
    outfile<<'\n';

    for(int row_index=0; row_index<N_rows; ++row_index){
        //y values
        outfile<<y_range[row_index];
        for(int col_index{0}; col_index<N_cols; ++col_index){
            //z values
            outfile<<","<<matrix(row_index, col_index);
        }
        outfile<<'\n';
    }
}

// t = 1
int main(int argc, char *argv[]){
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(7, 5, 2.4);
    std::vector<double> x_range = {1.1, 1.2, 1.3, 1.4, 1.5}; 
    std::vector<double> y_range = {1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3}; 
    write_matrix_to_CSV("test.txt", matrix, x_range, y_range);
}


