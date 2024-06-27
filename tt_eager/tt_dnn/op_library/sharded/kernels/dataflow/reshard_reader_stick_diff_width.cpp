// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"


void kernel_main() {
	constexpr uint32_t shard_cb = get_compile_time_arg_val(0);
	constexpr uint32_t num_x_cores = get_compile_time_arg_val(1);
	constexpr uint32_t num_y_cores = get_compile_time_arg_val(2);
	constexpr uint32_t page_size = get_compile_time_arg_val(3);
	constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
	constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
	constexpr uint32_t input_page_allignment = get_compile_time_arg_val(6);
	constexpr uint32_t output_page_allignment = get_compile_time_arg_val(7);
	constexpr uint32_t temp_cb = get_compile_time_arg_val(8);

	uint32_t y_offset = num_x_cores;

	uint32_t arg_index = num_x_cores + num_y_cores;
	const uint32_t input_shard_addr  = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_output_pages = get_arg_val<uint32_t>(arg_index++);
	const uint32_t num_ranges = get_arg_val<uint32_t>(arg_index++);
	const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_index++);

	uint32_t l1_write_base_addr = get_write_ptr(shard_cb);
	uint32_t l1_write_addr = l1_write_base_addr + output_page_offset * (page_size + output_page_allignment);

	uint32_t mask_byte = 0x0ff; //8 bits
	uint32_t mask_short = 0x0ffff; //16 bits

	uint32_t scratch_pad_base_addr = get_write_ptr(temp_cb);


	for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
		const uint32_t core_start_stride = get_arg_val<uint32_t>(arg_index++);
		const uint32_t start_x_index = (core_start_stride >> 24);
		const uint32_t start_y_index = (core_start_stride >> 16) & mask_byte;
		const uint32_t stride_x = (core_start_stride >> 8) & mask_byte;
		const uint32_t stride_y = (core_start_stride) & mask_byte;
		const uint32_t start_x = get_arg_val<uint32_t>(start_x_index);
		const uint32_t start_y = get_arg_val<uint32_t>(y_offset + start_y_index);

		const uint32_t stride_data_offset = get_arg_val<uint32_t>(arg_index++);
		const uint32_t stride_size_num_strides_skip = get_arg_val<uint32_t>(arg_index++);
		const uint32_t num_strides = ((stride_size_num_strides_skip) & mask_short) >> 8;
		const bool skip = (((stride_size_num_strides_skip) & mask_byte)  == 1);


		const uint32_t stride_data = ((stride_data_offset >> 16)) * page_size;

		uint32_t input_addr_offset = ((stride_data_offset) & mask_short) * (page_size + input_page_allignment);
		const uint32_t num_pages_per_stride = (stride_size_num_strides_skip >> 16);
		const uint32_t stride_size = num_pages_per_stride * page_size;

		uint32_t core_id_x_index = start_x_index;
		uint32_t core_id_y_index = start_y_index;

		uint32_t scratch_pad_addr = scratch_pad_base_addr;
		if(!skip) {
			uint32_t num_input_iterations = num_strides;
			uint32_t total_data_read = 0;

			//Reads input stride at a time into scratchpad
			for(uint32_t stride_idx = 0; stride_idx < num_input_iterations ; stride_idx++) {
				uint32_t core_id_x = get_arg_val<uint32_t>(core_id_x_index);
				uint32_t core_id_y = get_arg_val<uint32_t>(y_offset + core_id_y_index);
				uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
						input_shard_addr + input_addr_offset);
				noc_async_read(noc_address, scratch_pad_addr, stride_size);
				noc_async_read_barrier();
				scratch_pad_addr += stride_size;

				if(stride_x == 0 and stride_y == 0) {
					input_addr_offset += (stride_data + stride_size + input_page_allignment);
				}
				else {
					input_addr_offset += (stride_data);
				}
				core_id_x_index += stride_x;
				core_id_y_index += stride_y;
				total_data_read+=stride_size;
			}
			noc_async_read_barrier();


			//At this point entire shard is in scratchpad
			uint32_t num_output_pages_in_range = total_data_read/output_page_size;
			scratch_pad_addr = scratch_pad_base_addr;
			uint32_t num_output_iterations = num_output_pages_in_range;
			//writes output from scratchpad , output row at a time
			for(uint32_t stride_idx = 0; stride_idx < num_output_iterations; stride_idx++) {
				//local copy from scratchpad to output shard
				noc_async_read(get_noc_addr(scratch_pad_addr), l1_write_addr, output_page_size);
				l1_write_addr += (output_page_size + output_page_allignment);
				scratch_pad_addr += output_page_size;
			}
			noc_async_read_barrier();
		}

	}
	noc_async_read_barrier();

}
