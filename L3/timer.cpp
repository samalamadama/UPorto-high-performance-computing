#include <iostream>
#include <chrono>

int main(int arc, char *argv[]){
    int number_of_threads = atoi(argv[0]);
    auto start = std::chrono::system_clock::now();

    int result = system(argv[1]);

    auto end = std::chrono::system_clock::now();
    auto milliseconds_spent = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<result<<'\n';
    std::cout<<milliseconds_spent;
}