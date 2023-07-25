#pragma once

#include <atomic>

namespace tt::tt_metal::detail{

/**
 * Enable kernel compilation cache to be persistent across runs. When this is called, kernels will not be compiled if the output binary path exists.
 *
 * Return value: void
 */
void EnableCompileCache();

/**
 * Disables kernel compilation cache from being persistent across runs.
 *
 * Return value: void
 */
void DisableCompileCache();

/**
 * Returns bool indicating whether persistent caching is enabled.
 *
 * Return value: bool
 */
bool GetCompileCacheEnabled();

}
