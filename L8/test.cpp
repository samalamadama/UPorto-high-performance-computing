#include "/home/adthor/Desktop/QUARMEN/HPC/library_HPC/benchmarker.hpp"
#include <chrono>
#include <fstream>
#include <omp.h>

int main() {
  for (int thread_number{1}; thread_number != 9; ++thread_number) {
    omp_set_num_threads(thread_number);
    HPC::Benchmarker<std::string> benchmarker("test begins");
#pragma omp parallel for
    for (int i = 0; i != 512; ++i) {
    }
    benchmarker.save_time("test ends");
    std::ofstream benchmark_file("exercise_9_data/test_benchmark_" +
                                 std::to_string(thread_number) + "_threads");
    benchmarker.print_time_intervals<std::chrono::nanoseconds>(benchmark_file);
  }
}