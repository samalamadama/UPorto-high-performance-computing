#include <cmath>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include "fftw3.h"
#include <omp.h>

template <typename Derived>
Eigen::VectorXcd fft(const Eigen::MatrixBase<Derived>& input, bool forward = true) {
    int N = input.size();
    static thread_local fftw_plan plan;
    static thread_local fftw_complex* fftw_in = nullptr;
    static thread_local fftw_complex* fftw_out = nullptr;

    // Allocate FFTW input/output arrays only once per thread
    if (!fftw_in) {
        fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        fftw_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;
        plan = fftw_plan_dft_1d(N, fftw_in, fftw_out, sign, FFTW_ESTIMATE);
    }

    // Copy data to FFTW input buffer
    for (int i = 0; i < N; ++i) {
        fftw_in[i][0] = input[i].real();
        fftw_in[i][1] = input[i].imag();
    }

    // Execute FFT
    fftw_execute(plan);

    // Store results in Eigen vector
    Eigen::VectorXcd output(N);
    for (int i = 0; i < N; ++i) {
        output[i] = {fftw_out[i][0], fftw_out[i][1]};
    }

    return output;
}


// t = 1, periodic boundary conditions
int main(int argc, char *argv[]){
    int number_of_threads = 16;
    if(argc>1){
        number_of_threads = (int)argv[1][0]-(int)0;
        }
    std::ofstream outfile_direct;
    std::ofstream outfile_reciprocal;

    outfile_direct.open("n_evolution");
    outfile_reciprocal.open("k_evolution");

    double final_time = 1000000;
    int const T_divisions = 1024;
    int const N = 1024;
    float const n_0 = static_cast<float>(N)/2;
    double k_0 {1.};
    double stddev = static_cast<float>(N)/16;

    //omp_set_num_threads(number_of_threads);
    // fftw_init_threads();
    // fftw_plan_with_nthreads(4);

    auto psi_n_coefficients = [=](int n){ 
        std::complex<double> value = std::exp(-std::complex(std::pow((n-n_0)/stddev, 2)/2, k_0*n));
        return value;
        };
    std::array<double, N> n_range;
    std::generate(n_range.begin(), n_range.end(), [x = 0, dx=1] () mutable { 
        return x+=dx; 
    });
    //double normalization = std::transform_reduce(n_range.begin(), n_range.end(), 0, std::plus<>(), psi_n_coefficients);
    
    auto energy_function_pbc = [=, t=1](int a){return -2.0*t*std::cos(2*M_PI*a/N);};

    std::array<double, T_divisions> t_range;
    std::generate(t_range.begin(), t_range.end(), [t = 0., dt=final_time/T_divisions] () mutable { 
        return t+=dt; 
    });

    Eigen::VectorXcd psi_0_n = Eigen::VectorXcd::Zero(N);
    std::transform(n_range.begin(), n_range.end(), psi_0_n.data(), psi_n_coefficients);
    Eigen::VectorXcd psi_0_k = fft(psi_0_n);

    Eigen::MatrixXcd direct_space_time_ev=Eigen::MatrixXcd::Zero(T_divisions, N);
    Eigen::MatrixXcd reciprocal_space_time_ev=Eigen::MatrixXcd::Zero(T_divisions, N);


    #pragma omp declare reduction (+: Eigen::MatrixXcd: omp_out=omp_out+omp_in)\
     initializer(omp_priv=Eigen::MatrixXcd::Zero(T_divisions, N))

    #pragma omp parallel for schedule(static, 1) reduction(+:reciprocal_space_time_ev)
    for(int t=0; t<T_divisions;++t){
        for(int k=0; k<N; ++k){
            reciprocal_space_time_ev(t, k) = psi_0_k[k]*std::exp(std::complex(0., energy_function_pbc(k)*t_range[t]));
        }
    }

    #pragma omp parallel for schedule(static, 1) reduction(+:direct_space_time_ev)
    for(int t=0; t<T_divisions;++t){
        direct_space_time_ev.row(t) = fft(reciprocal_space_time_ev.row(t).transpose(), 0).transpose();
    }

  
    fftw_cleanup_threads();
    //print output files, it's in the form 
    //  0.0 x_1     x_2     x_3     ...     x_n
    //  y_1 z_11    z_12    ...             z_1n
    //  y_2 ...             ...             ...
    //  ... 
    //  y_N z_N1    ...                     z_Nn

    //x line of the output
    outfile_direct<<0.0000000<<"    ";
    outfile_reciprocal<<0.0000000<<"    ";

    for(int i=0; i<N; ++i){
        outfile_direct<<n_range[i]<<"    ";
        outfile_reciprocal<<2*M_PI*n_range[i]/N<<"    ";
    }
    outfile_direct<<'\n';
    outfile_reciprocal<<'\n';


    for(int j=0; j<T_divisions; ++j){
        //y values
        outfile_direct<<t_range[j]<<"    ";
        outfile_reciprocal<<t_range[j]<<"    ";
        for(int l{0}; l<N; ++l){
            //z values
            outfile_direct<<"    "<<std::abs(direct_space_time_ev(j, l))/N;
            outfile_reciprocal<<"    "<<std::abs(reciprocal_space_time_ev(j, l))/std::pow(N, 0.5);
        }
        outfile_direct<<'\n';
        outfile_reciprocal<<'\n';
    }
}