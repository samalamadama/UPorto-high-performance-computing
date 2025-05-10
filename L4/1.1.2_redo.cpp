#include <cmath>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <omp.h>
#include <vector>

#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/csv_writer.hpp"


// t = 1
int main(int argc, char *argv[]){
    double stddev{0.1};
    int const N = 2048;
    int const n = 2048;

    std::vector<double> E_range(n);
    std::generate(E_range.begin(), E_range.end(), [x = -3.0, dx=6.0/n] () mutable { 
        return x+=dx; 
    }); 
    std::vector<double> n_range(N);
    std::generate(n_range.begin(), n_range.end(), [x = 0, dx=1] () mutable { 
        return x+=dx; 
    });

    Eigen::MatrixXd total_function_obc=Eigen::MatrixXd::Zero(n, N);
    Eigen::MatrixXd total_function_pbc = Eigen::MatrixXd::Zero(n, N);

    int number_of_threads = 8;
    //if a number of thread is passed to function, use that
    if(argc>1){
        number_of_threads = (int)argv[1][0]-(int)'0';
        }
    omp_set_num_threads(number_of_threads);

    #pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in)\
    initializer(omp_priv=Eigen::MatrixXd::Zero(n,N))
    #pragma omp parallel firstprivate(n_range, E_range, n, N)
    { 
    auto energy_function_obc = [=, t = 1](int m){return -2.0*t*std::cos(M_PI*m/(N+1));};
    auto energy_function_pbc = [=, t=1](int a){return -2.0*t*std::cos(2*M_PI*a/N);};
    auto psi_n_pbc = [N](double alpha, int pos){
        std::complex<double> psi_unnormalized = std::exp(std::complex<double>(0., 2.*M_PI*alpha/N*pos));
        return psi_unnormalized/std::pow(N, 0.5); 
        };
    auto psi_n_obc = [N](int alpha, int pos){
        double psi_unnormalized = std::sin(M_PI*static_cast<double>(alpha)*(pos+1)/(N+1));
        return psi_unnormalized*std::pow(2./(N+1), 0.5);
        };
    auto lorentzian = [stddev](double mode, double x){
        return stddev/(M_PI*(std::pow(x-mode,2)+std::pow(stddev, 2)));
    };

    #pragma omp for schedule(dynamic) //reduction(+:total_function_obc, total_function_pbc)
    for(int E=0; E<n;++E){
        Eigen::VectorXd result_pbc(N);
        Eigen::VectorXd result_obc(N);
        for(int pos=0; pos<N; ++pos){
            for(int alpha=0; alpha<N; ++alpha){
                result_pbc(pos) += lorentzian(energy_function_pbc(alpha), E_range[E])*std::pow(std::abs(psi_n_pbc(alpha, pos)), 2);
                result_obc(pos) += lorentzian(energy_function_obc(alpha), E_range[E])*std::pow(psi_n_obc(alpha, pos),2);
            }
        }
        total_function_obc.row(E) = result_obc.transpose();
        total_function_pbc.row(E) = result_pbc.transpose();
    }
    }
    write_matrix_to_CSV("Energy_density_OBC", total_function_obc, n_range, E_range);
    write_matrix_to_CSV("Energy_density_PBC", total_function_pbc, n_range, E_range);
}