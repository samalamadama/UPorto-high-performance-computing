#include <iostream>
#include <omp.h>
#include <unistd.h>

int main(){
    double sum{0};
    omp_set_num_threads(4);
    #pragma omp parallel reduction(+:sum)
    {
        sum = omp_get_thread_num();
        #pragma omp critical
        {
            std::cout <<"local variable:" <<sum<< std::endl;
        }
    }
    std::cout<<"sum = "<<sum<<'\n';
}
