/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_cluster.hpp"

namespace tt {
namespace llrt {

void watcher_attach(void *dev, int device_id, const std::function<CoreCoord ()>& get_grid_size, const std::function<CoreCoord (CoreCoord)>& worker_from_logical, const std::function<const std::set<CoreCoord> &()>& storage_only_cores, const string& log_path);
void watcher_detach(void *dev);

void watcher_sanitize_host_noc_read(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);
void watcher_sanitize_host_noc_write(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);

} // namespace llrt
} // namespace tt
