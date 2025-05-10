#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <thread>

int main(){
    unsigned count[4] = {0};
    omp_set_num_threads(4);
    #pragma omp parallel shared(count)
    {
        unsigned th_id = omp_get_thread_num();
        #pragma omp for schedule(static, 20) collapse(1)
        for(unsigned i=0; i<11; i++)
            for(unsigned j=0; j<11; j++)
                for(unsigned k=0; k<11; k++){
                    std::this_thread::sleep_for(std::chrono::milliseconds(10*(th_id+1)));
                    count[th_id]++;
                }
    }
    for(int i=0; i<4; ++i){
        std::cout<<count[i]<<'\n';
    }
}