#include <omp.h>
#include <iostream>

int main(int arc, char *argv[]){
    unsigned M = atoi(argv[1]);
    double x = atof(argv[2]);

    omp_set_num_threads(2)
    #pragma omp parallel if(M>10)
    {
        #pragma omp critical{
            std::cout<<M<<'\n';
            std::cout<<x<<'\n';
            std::cout<<omp_get_num_threads()<<'\n';
            std::cout<<omp_get_thread_num()<<'\n';
        }
    }
}