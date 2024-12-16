// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>

#include "impl/device/device.hpp"

// Forward Declaration to avoid trace_buffer.hpp
namespace tt::tt_metal::detail {
    class TraceDescriptor;
}

// Forward decl for command_generated.h
namespace tt::target {
    struct Command;
    struct ReplayTraceCommand;
    struct EnqueueTraceCommand;
    struct LoadTraceCommand;
    struct ReleaseTraceCommand;
    struct CreateBufferCommand;
    struct DeallocateBufferCommand;
    struct EnqueueWriteBufferCommand;
    struct EnqueueReadBufferCommand;
    struct FinishCommand;
    struct CreateProgramCommand;
    struct EnqueueProgramCommand;
    struct CreateKernelCommand;
    struct SetRuntimeArgsCommand;
    struct SetRuntimeArgs1Command;
    struct CreateCircularBufferCommand;
    struct RuntimeArg;

    // Forward decl for binary_generated.h
    namespace lightmetal {
        struct TraceDescriptor;
        struct TraceDescriptorByTraceId;
        struct LightMetalBinary;
    }
}

using FlatbufferRuntimeArgVector = const flatbuffers::Vector<flatbuffers::Offset<tt::target::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<Buffer *, uint32_t>>;

namespace tt::tt_metal {
inline namespace v0 {

// General utility function to open a binary file and return contents as a binary blob
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob);

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob
    explicit LightMetalReplay(std::vector<uint8_t> blob);

    // Open a FlatBuffer binary from the stored blob
    const target::lightmetal::LightMetalBinary* openFlatBufferBinary();

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<detail::TraceDescriptor> getTraceByTraceId(uint32_t target_trace_id);

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> fromFlatBufferRtArgs(const FlatbufferRuntimeArgVector flatbuffer_args);

    // Execute the stored LightMetal binary
    bool executeLightMetalBinary();

    // Executor functions for all traced host API calls
    void execute(tt::target::Command const *command);
    void execute(tt::target::EnqueueTraceCommand const *command);
    void execute(tt::target::ReplayTraceCommand const *command);
    void execute(tt::target::LoadTraceCommand const *command);
    void execute(tt::target::ReleaseTraceCommand const *command);
    void execute(tt::target::CreateBufferCommand const *command);
    void execute(tt::target::DeallocateBufferCommand const *command);
    void execute(tt::target::EnqueueWriteBufferCommand const *command);
    void execute(tt::target::EnqueueReadBufferCommand const *command);
    void execute(tt::target::FinishCommand const *command);
    void execute(tt::target::CreateProgramCommand const *command);
    void execute(tt::target::EnqueueProgramCommand const *command);
    void execute(tt::target::CreateKernelCommand const *command);
    void execute(tt::target::SetRuntimeArgsCommand const *command);
    void execute(tt::target::SetRuntimeArgs1Command const *command);
    void execute(tt::target::CreateCircularBufferCommand const *command);

    // Object maps public accessors
    void addBufferToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Buffer> buffer);
    std::shared_ptr<::tt::tt_metal::Buffer> getBufferFromMap(uint32_t global_id) const;
    void removeBufferFromMap(uint32_t global_id);

    void addProgramToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Program> program);
    std::shared_ptr<::tt::tt_metal::Program> getProgramFromMap(uint32_t global_id) const;
    void removeProgramFromMap(uint32_t global_id);

    void addKernelHandleToMap(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id);
    ::tt::tt_metal::KernelHandle getKernelHandleFromMap(uint32_t global_id) const;
    void removeKernelHandleFromMap(uint32_t global_id);

    void addKernelToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Kernel> kernel);
    std::shared_ptr<::tt::tt_metal::Kernel> getKernelFromMap(uint32_t global_id) const;
    void removeKernelFromMap(uint32_t global_id);

    void addCBHandleToMap(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle);
    ::tt::tt_metal::CBHandle getCBHandleFromMap(uint32_t global_id) const;
    void removeCBHandleFromMap(uint32_t global_id);

private:

    // Workload related members --------------------
    const target::lightmetal::LightMetalBinary* parseFlatBufferBinary();

    std::vector<uint8_t> blob_;  // Stored binary blob
    const target::lightmetal::LightMetalBinary* lm_binary_;  // Parsed FlatBuffer binary

    // System related members ----------------------
    void setupDevices();

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;

    // Object maps for storing objects by global_id
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> bufferMap_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> programMap_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernelHandleMap_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernelMap_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cbHandleMap_;
};

}  // namespace v0
}  // namespace tt::tt_metal
