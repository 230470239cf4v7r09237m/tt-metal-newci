#pragma once

// NOTE: this ifdef must be **exactly aligned** with ifdef in kernel_profiler.hpp, or
// timeStampedData will be fwd declared and used, but then never defined.
#if defined(PROFILE_NOC_EVENTS) && defined(PROFILE_KERNEL) && ( !defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && defined(COMPILE_FOR_NCRISC) && (PROFILE_KERNEL == PROFILER_OPT_DO_DISPATCH_CORES)))

#include "event_metadata.hpp"
#include "risc_attribs.h"
#include "kernel_profiler_fwd.hpp"

namespace noc_event_profiler {

inline std::pair<uint32_t, uint32_t> decode_noc_xy_to_coord(uint32_t noc_xy) {
    // shift so that coordinate is in LSB
    noc_xy = noc_xy >> NOC_COORD_REG_OFFSET;

    constexpr uint32_t NOC_COORD_MASK = 0x3F;

    uint32_t encoded_x = (noc_xy)&NOC_COORD_MASK;
    uint32_t decoded_x = (noc_index == 1) ? (noc_size_x - 1 - encoded_x) : encoded_x;

    uint32_t encoded_y = (noc_xy >> (NOC_ADDR_NODE_ID_BITS)) & NOC_COORD_MASK;
    uint32_t decoded_y = (noc_index == 1) ? (noc_size_y - 1 - encoded_y) : encoded_y;

    return {decoded_x, decoded_y};
}

inline std::pair<uint32_t, uint32_t> decode_noc_addr_to_coord(uint64_t noc_addr) {
    // See noc_parameters.h for definition of NOC address
    constexpr int NOC_COORD_MASK = 0x3F;

    uint32_t encoded_x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & NOC_COORD_MASK;
    uint32_t decoded_x = (noc_index == 1) ? (noc_size_x - 1 - encoded_x) : encoded_x;

    uint32_t encoded_y = (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & NOC_COORD_MASK;
    uint32_t decoded_y = (noc_index == 1) ? (noc_size_y - 1 - encoded_y) : encoded_y;

    return {decoded_x, decoded_y};
}

template <bool DRAM>
inline std::pair<uint32_t, uint32_t> decode_noc_id_into_coord(uint32_t id, uint8_t noc = noc_index) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
    return decode_noc_xy_to_coord(interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc));
}

template <uint32_t STATIC_ID = 12345>
inline void recordNocEvent(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type,
    uint32_t dst_x = 0,
    uint32_t dst_y = 0,
    uint32_t num_bytes = 0,
    uint8_t noc = noc_index) {
    KernelProfilerNocEventMetadata ev_md;
    ev_md.dst_x = dst_x;
    ev_md.dst_y = dst_y;
    ev_md.noc_xfer_type = noc_event_type;
    ev_md.num_bytes = num_bytes;
    ev_md.noc_type =
        (noc == 1) ? KernelProfilerNocEventMetadata::NocType::NOC_1 : KernelProfilerNocEventMetadata::NocType::NOC_0;

#if defined(DISPATCH_KERNEL)
#define DISPATCH true
#else 
#define DISPATCH false
#endif
    kernel_profiler::timeStampedData<STATIC_ID,DISPATCH>(ev_md.asU64());
}

// for noc_async_(read|write)_tile functions
template <bool DRAM, typename NocIDU32>
inline void recordNocEventWithID(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type, NocIDU32 noc_id, uint32_t num_bytes) {
    // check that first argument is NOT a 64b noc addr by mistake
    static_assert(std::is_same_v<NocIDU32, uint32_t>);
    auto [decoded_x, decoded_y] = decode_noc_id_into_coord<DRAM>(noc_id);
    recordNocEvent(noc_event_type, decoded_x, decoded_y, num_bytes);
}

// for noc_async_(read|write) functions
template <typename NocAddrU64>
inline void recordNocEventWithAddr(
    KernelProfilerNocEventMetadata::NocEventType noc_event_type, NocAddrU64 noc_addr, uint32_t num_bytes) {
    // check that first argument is NOT a 32b noc id by mistake
    static_assert(std::is_same_v<NocAddrU64, uint64_t>);
    auto [decoded_x, decoded_y] = decode_noc_addr_to_coord(noc_addr);
    recordNocEvent(noc_event_type, decoded_x, decoded_y, num_bytes);
}

}  // namespace noc_event_profiler

#define RECORD_NOC_WRITE_EVENT_WITH_ADDR(noc_addr, num_bytes)                          \
    if constexpr (enable_noc_tracing) {                                                \
        noc_event_profiler::recordNocEventWithAddr(                                    \
            KernelProfilerNocEventMetadata::NocEventType::WRITE, noc_addr, num_bytes); \
    }

#define RECORD_NOC_WRITE_EVENT_WITH_ID(noc_id, num_bytes)                            \
    if constexpr (enable_noc_tracing) {                                              \
        noc_event_profiler::recordNocEventWithID<DRAM>(                              \
            KernelProfilerNocEventMetadata::NocEventType::WRITE, noc_id, num_bytes); \
    }

#define RECORD_NOC_READ_EVENT_WITH_ADDR(noc_addr, num_bytes)                          \
    if constexpr (enable_noc_tracing) {                                               \
        noc_event_profiler::recordNocEventWithAddr(                                   \
            KernelProfilerNocEventMetadata::NocEventType::READ, noc_addr, num_bytes); \
    }
#define RECORD_NOC_READ_EVENT_WITH_ID(noc_id, num_bytes)                            \
    if constexpr (enable_noc_tracing) {                                             \
        noc_event_profiler::recordNocEventWithID<DRAM>(                             \
            KernelProfilerNocEventMetadata::NocEventType::READ, noc_id, num_bytes); \
    }

#define RECORD_NOC_WRITE_BARRIER()                                                                       \
    if constexpr (enable_noc_tracing) {                                                                  \
        noc_event_profiler::recordNocEvent(KernelProfilerNocEventMetadata::NocEventType::WRITE_BARRIER); \
    }
#define RECORD_NOC_READ_BARRIER()                                                                       \
    if constexpr (enable_noc_tracing) {                                                                 \
        noc_event_profiler::recordNocEvent(KernelProfilerNocEventMetadata::NocEventType::READ_BARRIER); \
    }

#else

#define RECORD_NOC_READ_EVENT_WITH_ADDR(noc_addr, num_bytes)
#define RECORD_NOC_WRITE_EVENT_WITH_ADDR(noc_addr, num_bytes)

#define RECORD_NOC_READ_EVENT_WITH_ID(noc_id, num_bytes)
#define RECORD_NOC_WRITE_EVENT_WITH_ID(noc_id, num_bytes)

#define RECORD_NOC_WRITE_BARRIER()
#define RECORD_NOC_READ_BARRIER()

#endif
