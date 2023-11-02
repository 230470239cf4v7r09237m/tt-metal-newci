/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "common_globals.h"


namespace ckernel {

/**
 * Helper function to reconfigure unpacker srca and srcb input data formats.
 */
ALWI void unpack_reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format(srca_new_operand, srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

ALWI void unpack_reconfig_data_format(const uint32_t srca_old_operand, const uint32_t srca_new_operand, const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

ALWI void unpack_reconfig_data_format_srca(const uint32_t srca_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format_srca(srca_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

ALWI void unpack_reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format_srca(srca_old_operand, srca_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

ALWI void unpack_reconfig_data_format_srcb(const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format_srcb(srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

ALWI void unpack_reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand) {
    #ifdef ARCH_GRAYSKULL
        UNPACK(( llk_unpack_reconfig_data_format_srcb(srcb_old_operand, srcb_new_operand) ));
    #endif
    // NOTE: For wormhole_b0, updated unpacker functions don't yet exist, so skip.
}

}
