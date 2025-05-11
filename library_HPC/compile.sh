
if [ "$2" = "fast" ]; then 
    g++ "$1".cpp -o "$1" -O2
else 
    g++ "$1".cpp -o "$1" -Wall -Wextra -Wpedantic -Wcast-align -Wcast-qual -Wunused-function -Wdisabled-optimization -Wduplicated-branches -Wduplicated-cond -Wformat=2 -Wlogical-op -Wmissing-include-dirs -Wnull-dereference -Woverloaded-virtual -Wpointer-arith -Wshadow -Wswitch-enum -Wvla -fopenmp -lfftw3 -lfftw3_threads -fsanitize=address -fno-omit-frame-pointer -lm -pedantic-errors -march=native -std=c++23 -O2
fi




