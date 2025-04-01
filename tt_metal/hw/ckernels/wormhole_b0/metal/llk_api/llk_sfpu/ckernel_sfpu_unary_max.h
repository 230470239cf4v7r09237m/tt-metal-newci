// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_unary_max(uint value) {
    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat scalar = c_value.f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input_val = dst_reg[0];
        v_if(input_val < scalar) { input_val = scalar; }
        v_endif;
        dst_reg[0] = input_val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
