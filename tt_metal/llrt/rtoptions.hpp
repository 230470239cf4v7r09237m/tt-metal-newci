/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

namespace tt {

namespace llrt {

class RunTimeOptions {
    int watcher_interval_ms;
    bool watcher_dump_all;

public:
    RunTimeOptions();

    inline bool get_watcher_enabled() { return watcher_interval_ms != 0; }
    inline int get_watcher_interval() { return watcher_interval_ms; }
    inline int get_watcher_dump_all() { return watcher_dump_all; }
};


extern RunTimeOptions OptionsG;

} // namespace llrt

} // namespace tt
