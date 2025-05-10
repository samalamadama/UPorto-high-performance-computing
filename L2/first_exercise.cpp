#include <cmath>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

class Gaussian{
    private:
    double const _mean;
    double const _stddev;

    public:
    Gaussian(double mean, double stddev) : _mean(mean), _stddev(stddev) {}
    double operator()(double x)const {
        double coeff{1.0/(_stddev*std::sqrt(M_PI*2))};
        double expo{std::exp(-0.5*std::pow((x-_mean)/_stddev, 2))};
        return coeff*expo;
    }
}; 

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

class Square_interval{
    private:
    double const _mean;
    double const _width;

    public:
    Square_interval(double mean, double width) : _mean(mean), _width(width){}
    double operator()(double x) const {
        return (std::abs(x-_mean)<_width/2 ? 1/_width : 0);
    }
};

// t = 1
int main(){
    std::ofstream outfile;
    outfile.open("Energy density");
    double stddev{0.001};
    int const n = 100000;
    int N = 130658;
    auto energy_function = [=, t=1](int a){return -2.0*t*std::cos(2*M_PI*a/N);};
    //generates the array "range" as a set of n values equispaced between -2 and 2 (domain of our cosine)
    std::array<double, n> range;
    std::generate(range.begin(), range.end(), [x = -3.0, dx=6.0/n] () mutable { 
        return x+=dx; 
    }); 
    double total_function[n]{};

    #pragma omp parallel for schedule(static, 1) reduction(+:total_function[:n])
        for(int l=0; l<=N; ++l){
            double mean{energy_function(l)};
            Gaussian delta (mean, stddev);
            double evaluated_function[n];
            std::transform(range.begin(), range.end(), evaluated_function, delta);
            std::transform(evaluated_function, evaluated_function+n, total_function, total_function, std::plus<double>());
        }
    for(int i{0}; i<n; ++i){
        outfile<<range[i]<<"    "<<total_function[i]<<'\n';
    }

}