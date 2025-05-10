#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#define M 100000000

int main(){
    std::random_device r;
    std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 e2(seed2);
    std::uniform_real_distribution<> dist;
    unsigned count{0};
    auto start = std::chrono::system_clock::now();
    #pragma omp parallel reduction(+:count) firstprivate(dist, e2)
    {
    int num_threads = omp_get_num_threads();
    #pragma omp for schedule(static, M/num_threads)
        for(int i=0; i<M; i++){
            double x = dist(e2);
            double y = dist(e2);
            count += (x*x+y*y <1 ? 1 : 0);
        }
    }
    auto end = std::chrono::system_clock::now();
    auto milliseconds_spent = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<4*(count*1.0/M) <<'\n' <<milliseconds_spent<<std::endl;
    return 0;
}