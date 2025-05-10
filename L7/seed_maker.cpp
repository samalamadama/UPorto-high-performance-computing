#include <iostream>
#include <fstream>
#include <random>

int main() {
    constexpr int num_seeds = 1000;
    constexpr int min_seed = 0;
    constexpr int max_seed = std::numeric_limits<int>::max();

    std::ofstream seed_file("seeds.txt");
    if (!seed_file) {
        std::cerr << "Error: Could not open seeds.txt for writing.\n";
        return 1;
    }

    std::random_device rd;  // Non-deterministic seed source
    std::mt19937 generator(rd());  // Mersenne Twister engine
    std::uniform_int_distribution<int> distribution(min_seed, max_seed);

    for (int i = 0; i < num_seeds; ++i) {
        seed_file << distribution(generator) << '\n';
    }

    std::cout << "Generated " << num_seeds << " seeds in seeds.txt\n";
    return 0;
}