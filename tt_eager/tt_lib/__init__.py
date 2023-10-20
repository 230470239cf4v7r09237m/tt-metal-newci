# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from ._C import tensor, device, dtx, profiler, program_cache, operations
from . import fallback_ops


def empty(*args, **kwargs):
    return tensor.zeros(*args, **kwargs)


setattr(tensor, "empty", empty)
