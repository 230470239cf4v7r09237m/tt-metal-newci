// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include <tt-metalium/assert.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

template <>
struct fmt::formatter<ttnn::operations::binary_ng::Lowercase> : fmt::formatter<std::string_view> {
    auto format(const ttnn::operations::binary_ng::Lowercase& value, fmt::format_context& ctx) const {
        auto out = ctx.out();
        for (char c : value.view) {
            *out++ = std::tolower(static_cast<unsigned char>(c));
        }
        return out;
    }
};

namespace ttnn::operations::binary_ng {

BinaryNgKernelConfig::BinaryNgKernelConfig(SubtileBroadcastType subtile_broadcast_type) {
    switch (subtile_broadcast_type) {
        case SubtileBroadcastType::NONE:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::SCALAR_A:
            reader_kernel = KernelName::ReaderScalarBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::SCALAR_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterScalarBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::ROW_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeNoBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = std::nullopt;
            break;

        case SubtileBroadcastType::COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterNoBcast;
            bcast_input = 0;
            break;

        case SubtileBroadcastType::COL_B:
            reader_kernel = KernelName::ReaderNoBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_A_COL_B:
            reader_kernel = KernelName::ReaderRowBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterColBcast;
            bcast_input = 1;
            break;

        case SubtileBroadcastType::ROW_B_COL_A:
            reader_kernel = KernelName::ReaderColBcast;
            compute_kernel = KernelName::ComputeBcast;
            writer_kernel = KernelName::WriterRowBcast;
            bcast_input = 0;
            break;
    }
}

std::string BinaryNgKernelConfig::bcast_input_str() const {
    if (bcast_input.has_value()) {
        return std::to_string(*bcast_input);
    }
    return "";
}

std::string get_kernel_file_path(KernelName kernel_name, bool is_sfpu) {
    constexpr std::string_view root = "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels";
    constexpr std::string_view dataflow = "{}/dataflow/{}";
    constexpr std::string_view compute = "{}/compute/{}";

    switch (kernel_name) {
        case KernelName::ReaderNoBcast: return fmt::format(dataflow, root, "reader_interleaved_no_bcast.cpp");
        case KernelName::ReaderRowBcast: return fmt::format(dataflow, root, "reader_interleaved_row_bcast.cpp");
        case KernelName::ReaderColBcast: return fmt::format(dataflow, root, "reader_interleaved_col_bcast.cpp");
        case KernelName::ReaderScalarBcast: return fmt::format(dataflow, root, "reader_interleaved_scalar_bcast.cpp");
        case KernelName::WriterNoBcast: return fmt::format(dataflow, root, "writer_interleaved_no_bcast.cpp");
        case KernelName::WriterRowBcast: return fmt::format(dataflow, root, "writer_interleaved_row_bcast.cpp");
        case KernelName::WriterColBcast: return fmt::format(dataflow, root, "writer_interleaved_col_bcast.cpp");
        case KernelName::WriterScalarBcast: return fmt::format(dataflow, root, "writer_interleaved_scalar_bcast.cpp");
        case KernelName::WriterScalar: return fmt::format(dataflow, root, "writer_interleaved_scalar.cpp");
        case KernelName::ComputeNoBcast:
            return fmt::format(
                compute, root, is_sfpu ? "eltwise_binary_sfpu_no_bcast.cpp" : "eltwise_binary_no_bcast.cpp");
        case KernelName::ComputeBcast:
            return fmt::format(compute, root, is_sfpu ? "eltwise_binary_sfpu.cpp" : "eltwise_binary.cpp");
        case KernelName::ComputeScalar:
            return fmt::format(compute, root, is_sfpu ? "eltwise_binary_sfpu_scalar.cpp" : "eltwise_binary_scalar.cpp");
        default: __builtin_unreachable();  // GCC 12 doesn't compile even though we exhaustively match
    }
}

template <class EnumT>
OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<EnumT>) : binary_op(EnumT::SUB) {
    // bool is_sfpu_op = std::holds_alternative<SfpuBinaryOp>(binary_op);
    // binary_op = is_sfpu_op ? std::variant<FpuBinaryOp, SfpuBinaryOp>(SfpuBinaryOp::SUB)
    //                        : std::variant<FpuBinaryOp, SfpuBinaryOp>(FpuBinaryOp::SUB);
    switch (binary_op_type) {
        case BinaryOpType::ADD: binary_op = EnumT::ADD; break;
        case BinaryOpType::SUB: break;
        case BinaryOpType::MUL: binary_op = EnumT::MUL; break;
        case BinaryOpType::DIV:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::DIV;
            } else {
                binary_op = FpuBinaryOp::MUL;
                process_rhs = unary::UnaryOpType::RECIP;
            }
            break;
        case BinaryOpType::RSUB:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::RSUB;
            } else {
                binary_op = FpuBinaryOp::ADD;
                process_rhs = unary::UnaryOpType::NEG;
            }
            break;
        case BinaryOpType::GT: postprocess = unary::UnaryOpType::GTZ; break;
        case BinaryOpType::LT: postprocess = unary::UnaryOpType::LTZ; break;
        case BinaryOpType::GTE: postprocess = unary::UnaryOpType::GEZ; break;
        case BinaryOpType::LTE: postprocess = unary::UnaryOpType::LEZ; break;
        case BinaryOpType::EQ: postprocess = unary::UnaryOpType::EQZ; break;
        case BinaryOpType::NE: postprocess = unary::UnaryOpType::NEZ; break;
        case BinaryOpType::SQUARED_DIFFERENCE: postprocess = unary::UnaryOpType::SQUARE; break;
        case BinaryOpType::BIAS_GELU:
            binary_op = EnumT::ADD;
            postprocess = unary::UnaryOpType::GELU;
            break;
        case BinaryOpType::LOGICAL_AND:
            binary_op = EnumT::MUL;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        case BinaryOpType::LOGICAL_OR:
            binary_op = EnumT::ADD;
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            postprocess = unary::UnaryOpType::GTZ;
            break;
        case BinaryOpType::LOGICAL_XOR:
            process_lhs = unary::UnaryOpType::NEZ;
            process_rhs = unary::UnaryOpType::NEZ;
            postprocess = unary::UnaryOpType::NEZ;
            break;
        case BinaryOpType::LDEXP:
            binary_op = EnumT::MUL;
            process_rhs = unary::UnaryOpType::EXP2;
            break;
        case BinaryOpType::LOGADDEXP:
            binary_op = EnumT::ADD;
            process_lhs = unary::UnaryOpType::EXP;
            process_rhs = unary::UnaryOpType::EXP;
            postprocess = unary::UnaryOpType::LOG;
            break;
        case BinaryOpType::LOGADDEXP2:
            binary_op = EnumT::ADD;
            process_lhs = unary::UnaryOpType::EXP2;
            process_rhs = unary::UnaryOpType::EXP2;
            postprocess = unary::UnaryOpType::LOG2;
            break;
        case BinaryOpType::BITWISE_AND:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_AND;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::BITWISE_OR:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_OR;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::BITWISE_XOR:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::BITWISE_XOR;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::LEFT_SHIFT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::LEFT_SHIFT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::RIGHT_SHIFT:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::RIGHT_SHIFT;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        case BinaryOpType::POWER:
            if (is_sfpu_op()) {
                binary_op = SfpuBinaryOp::POWER;
            } else {
                TT_THROW("Unsupported binary op for FPU {}", binary_op_type);
            }
            break;
        default: TT_THROW("Unsupported binary op {}", binary_op_type);
    }
}

std::pair<std::string, std::string> OpConfig::get_sfpu_init_fn(DataType dtype) const {
    // if (std::holds_alternative<SfpuBinaryOp>(binary_op)) {
    if (is_sfpu_op()) {
        auto sfpu_binary_op = std::get<SfpuBinaryOp>(binary_op);
        switch (sfpu_binary_op) {
            case SfpuBinaryOp::ADD:
                if (dtype == DataType::INT32) {
                    return {"add_int32_tile_init();", "add_int32_tile"};
                } else {
                    return {"add_binary_tile_init();", "add_binary_tile"};
                }
            case SfpuBinaryOp::SUB: return {"sub_binary_tile_init();", "sub_binary_tile"};
            case SfpuBinaryOp::MUL: return {"mul_binary_tile_init();", "mul_binary_tile"};
            case SfpuBinaryOp::DIV: return {"div_binary_tile_init();", "div_binary_tile"};
            case SfpuBinaryOp::POWER: return {"power_binary_tile_init();", "power_binary_tile"};
            case SfpuBinaryOp::RSUB: return {"rsub_binary_tile_init();", "rsub_binary_tile"};
            case SfpuBinaryOp::LEFT_SHIFT: return {"binary_shift_tile_init();", "binary_left_shift_tile"};
            case SfpuBinaryOp::RIGHT_SHIFT: return {"binary_shift_tile_init();", "binary_right_shift_tile"};
            case SfpuBinaryOp::BITWISE_AND: return {"binary_bitwise_tile_init();", "and_binary_tile"};
            case SfpuBinaryOp::BITWISE_OR: return {"binary_bitwise_tile_init();", "or_binary_tile"};
            case SfpuBinaryOp::BITWISE_XOR: return {"binary_bitwise_tile_init();", "xor_binary_tile"};
            default: TT_THROW("Unsupported sfpu binary op {}", binary_op);
        }
    } else {
        TT_THROW("SfpuBinaryOp not found");
    }
}

std::map<std::string, std::string> OpConfig::as_defines(DataType dtype) const {
    std::map<std::string, std::string> defines;

    if (!is_sfpu_op()) {
        if (!std::holds_alternative<FpuBinaryOp>(binary_op)) {
            TT_THROW("FpuBinaryOp not found");
        }
        auto fpu_binary_op = std::get<FpuBinaryOp>(binary_op);
        auto binary_op_str = magic_enum::enum_name(fpu_binary_op);
        defines["BINARY_OP"] = fmt::format("{}_tiles", Lowercase{binary_op_str});
        defines["BINARY_OP_TYPE"] = fmt::format("EltwiseBinaryType::ELW{}", binary_op_str);
        return defines;
    } else {
        auto&& [tile_init, tile_fn] = get_sfpu_init_fn(dtype);
        defines["BINARY_SFPU_INIT"] = std::move(tile_init);
        defines["BINARY_SFPU_OP"] = std::move(tile_fn);
        return defines;
    }
}

void add_activation_defines(
    std::map<std::string, std::string>& defines,
    tt::stl::Span<const unary::UnaryOpType> activations,
    std::string_view operand) {
    auto prepend_separator = false;
    std::string process = "";

    for (auto& a : activations) {
        if (prepend_separator) {
            process += ';';
        }
        prepend_separator = true;
        process += fmt::format("PROCESS_ACTIVATION({}, i)", magic_enum::enum_name(a));
        unary::utils::update_macro_defines(a, defines);
    }

    defines[fmt::format("PROCESS_{}_ACTIVATIONS(i)", operand)] = process;
}

bool OpConfig::is_sfpu_op() const { return std::holds_alternative<SfpuBinaryOp>(binary_op); }

template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<FpuBinaryOp>);
template OpConfig::OpConfig(BinaryOpType binary_op_type, std::in_place_type_t<SfpuBinaryOp>);

}  // namespace ttnn::operations::binary_ng
