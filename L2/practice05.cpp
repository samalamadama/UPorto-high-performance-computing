#include <cmath>
#include <omp.h>
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
        return ((x-_mean)<_width/2 ? 1/_width : 0);
    }
};

// t = 1
int main(){
    double sum {0};
    #pragma omp parallel
    {
        double partial_sum {0};
        #pragma omp for schedule(static, 1)
        for(int l=0; l<=20; ++l){
            partial_sum += 2;
        }
        #pragma omp critical
        sum +=partial_sum;
    }
    std::cout<<sum<<'\n';

}