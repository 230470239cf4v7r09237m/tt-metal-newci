/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <math.h>

#include "common/assert.hpp"

namespace tt {

inline uint32_t div_up(uint32_t a, uint32_t b) {
    TT_ASSERT(b > 0);
    return static_cast<uint32_t>((a+b-1)/b);
}

inline uint32_t round_up(uint32_t a, uint32_t b) {
    return b*div_up(a, b);
}

inline std::uint32_t round_down(std::uint32_t a, std::uint32_t b) {
    return  a / b * b;
}

inline uint32_t positive_pow_of_2(uint32_t exponent) {
    TT_ASSERT(exponent >= 0 && exponent < 32);
    uint32_t result = 1;
    for (uint32_t current_exp = 0; current_exp < exponent; current_exp++) {
        result *= 2;
    }

    return result;
}

}  // namespace tt
