#include <iostream>
#include <omp.h>
#include <stdio.h>

int main(){
    int a{0};
    int tib{1};
    float r{1.0};

    #pragma omp parallel firstprivate(a, r) shared(a) num_threads(2)
    {
        printf("\nX1: a = %d tib = %d r = %f", a, tib, r);
        tib = omp_get_thread_num();
        printf("\nX1: a = %d tib = %d r = %f", a, tib, r);
        a=tib;
        printf("\nX1: a = %d tib = %d r = %f", a, tib, r);
        }   

    printf("\n a = %d \n", a);
}
