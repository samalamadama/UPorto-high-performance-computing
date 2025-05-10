#include <cmath>
#include <array>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <omp.h>

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

#pragma omp declare reduction (+: Eigen::MatrixXd: omp_out=omp_out+omp_in) initializer(omp_priv=Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

// t = 1
int main(int argc, char *argv[]){
    int number_of_threads = 8;
    if(argc>1){
        number_of_threads = atoi(argv[1]);
        }
    std::ofstream outfile_obc;
    outfile_obc.open("Energy_density_OBC");
    std::ofstream outfile_pbc;
    outfile_pbc.open("Energy_density_PBC");
    double stddev{0.1};

    int const N =  argc>2? atoi(argv[2]): 512;
    int const n = argc>2? atoi(argv[2]): 512;
    
    auto energy_function_pbc = [=, t=1](int a){return -2.0*t*std::cos(2*M_PI*a/N);};
    auto energy_function_obc = [=, t = 1](int m){return -2.0*t*std::cos(M_PI*m/(N+1));};
    std::vector<double> E_range(n);
    std::generate(E_range.begin(), E_range.end(), [x = -3.0, dx=6.0/n] () mutable { 
        return x+=dx; 
    }); 
    Eigen::MatrixXd total_function_obc=Eigen::MatrixXd::Zero(n, N);
    Eigen::MatrixXd total_function_pbc = Eigen::MatrixXd::Zero(n, N);

    omp_set_num_threads(number_of_threads);
    #pragma omp parallel for schedule(static, 1) firstprivate(energy_function_pbc, energy_function_obc, E_range) reduction(+:total_function_obc, total_function_pbc)
    for(int pos=0; pos<N; ++pos){
        auto psi_n_pbc = [N, pos](double alpha){
            std::complex<double> psi_unnormalized = std::exp(std::complex<double>(0., 2*M_PI*alpha/N*pos));
            return psi_unnormalized/std::pow(N, 0.5); 
            };
        auto psi_n_obc = [N, pos](int alpha){
            double psi_unnormalized = std::sin(M_PI*static_cast<double>(alpha)*(pos+1)/(N+1));
            return psi_unnormalized*std::pow(2./(N+1), 0.5);
            };
        for(int E=0; E<n;++E){
            for(int alpha=0; alpha<N; ++alpha){
                Lorentzian local_density_pbc(energy_function_pbc(alpha), stddev);
                Lorentzian local_density_obc(energy_function_obc(alpha), stddev);
                total_function_pbc(E, pos) += local_density_pbc(E_range[E])*std::pow(std::abs(psi_n_pbc(alpha)), 2);
                total_function_obc(E, pos) += local_density_obc(E_range[E])*std::pow(psi_n_obc(alpha),2);
            }
        }
    }


    //print output files, it's in the form 
    //  0.0 x_1     x_2     x_3     ...     x_n
    //  y_1 z_11    z_12    ...             z_1n
    //  y_2 ...             ...             ...
    //  ... 
    //  y_N z_N1    ...                     z_Nn

    //x line of the output
    outfile_obc<<0.0000000<<"    ";
    outfile_pbc<<0.0000000<<"    ";
    for(int i=0; i<N; ++i){
        outfile_obc<<i<<"    ";
        outfile_pbc<<i<<"    ";
    }
    outfile_obc<<'\n';
    outfile_pbc<<'\n';

    for(int j=0; j<n; ++j){
        //y values
        outfile_obc<<E_range[j]<<"    ";
        outfile_pbc<<E_range[j]<<"    ";
        for(int l{0}; l<N; ++l){
            //z values
            outfile_obc<<"    "<<total_function_obc(j ,l);
            outfile_pbc<<"    "<<total_function_pbc(j, l);
        }
        outfile_obc<<'\n';
        outfile_pbc<<'\n';
    }

}