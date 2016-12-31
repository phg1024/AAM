#pragma once

#include "common.h"

template <typename C>
std::vector<std::pair<int, typename C::value_type>> enumerate(const C& container) {
  std::vector<std::pair<int, typename C::value_type>> pairs;
  int idx = 0;
  for(auto it=container.begin(); it!=container.end(); ++it) {
    pairs.push_back(std::make_pair(idx++, *it));
  }
  return pairs;
}