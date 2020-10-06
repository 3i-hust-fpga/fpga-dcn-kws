#pragma once
#include <cmath>
#include <tuple>

constexpr auto kMaxNumGrids = 65535;

inline int GET_BLOCKS(const int N, const int NUM_THREADS) {
    return std::min(kMaxNumGrids, (N + NUM_THREADS - 1) / NUM_THREADS);
}

static int get_greatest_divisor_below_bound(int n, int bound) {
    for (int k = std::min(n, bound); k > 1; --k) {
        if (n % k == 0) {
            return k;
        }
    }
    return 1;
}
