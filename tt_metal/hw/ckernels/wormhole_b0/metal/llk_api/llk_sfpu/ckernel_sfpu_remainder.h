// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder(const uint value) {

    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat s = c_value.f;
    vFloat value_tmp = s;
    s = sfpi::abs(s);

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat v = sfpi::abs(val);

        vFloat recip = sfpu_reciprocal<APPROXIMATION_MODE ? 2 : 3,true>(s);
        vFloat quotient = v*recip;

        vInt tmp = float_to_int16(quotient);
        vFloat newquotient= int32_to_float(tmp);
        v_if (newquotient > quotient){
            newquotient = newquotient - 1;
        }
        v_endif;
        v = v - newquotient * s;

        v_if(v>=s){
            v = v - s;
        }
        v_endif;

        v_if(val<0 && v!=0){
            v = s - v;
        }
        v_endif;

        v_if(value_tmp<0 && v!=0){
            v = v + value_tmp;
        }
        v_endif;
        v = setsgn(v, value_tmp);
        v_if(s==0){
            v = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
