#pragma once
#include <cmath>
#include <tuple>

static int get_greatest_divisor_below_bound(int n, int bound) {
	for (int k = bound; k > 1; --k) {
		if (n % k == 0) {
			return k;
		}
	}
	return 1;
}
