#ifndef BENCHMARKER
#define BENCHMARKER

#include <chrono>
#include <ctime>
#include <ostream>
#include <utility>
#include <vector>

namespace HPC {
using time = std::chrono::steady_clock;
using time_point = std::chrono::steady_clock::time_point;

template <typename Information_Class> class Benchmarker {
  std::vector<std::pair<Information_Class, time_point>> data_storage;

public:
  Benchmarker(Information_Class info = Information_Class()) { save_time(info); }

  void save_time(const Information_Class &info) {
    time_point current_time = time::now();
    data_storage.push_back(std::make_pair(info, current_time));
  }

  void print_time_points(std::ostream &out_stream,
                         std::string separator = ",") const {
    for (const auto &data_point : data_storage) {
      out_stream << data_point.first << separator << data_point.second << '\n';
    }
  }

  template <typename time_units = std::chrono::milliseconds>
  void print_time_intervals(std::ostream &out_stream,
                            std::string separator = ",") const {
    for (auto iterator = data_storage.cbegin() + 1;
         iterator != data_storage.cend(); ++iterator) {
      auto time_difference = std::chrono::duration_cast<time_units>(
          iterator->second - (iterator - 1)->second);
      out_stream << iterator->first << separator << time_difference << '\n';
    }
  }
};

} // namespace HPC

#endif