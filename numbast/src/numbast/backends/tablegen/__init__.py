# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TableGen (MLIR dialect) emission backend.

Emits MLIR dialect definitions rather than Numba bindings. The registered
names keep the two apart: ``Backend: numba-cuda-mlir`` selects the backend
that emits Numba typing and lowering registries against an MLIR runtime, and
this one emits the MLIR dialect itself. Naming either of them ``mlir`` would
describe both.

This reads a parsed header and renders a buildable dialect: the type
vocabulary, the op definitions and the ``.td`` files holding them, the C access
shim the ops lower to, the lowering descriptor that drives the generic
conversion patterns, and the C++ and CMake scaffolding that turns all of it
into a loadable shared object. It is selectable as ``Backend: mlir-dialect``.
"""

from numbast.backends.tablegen.config import (
    OutParamPolicy,
    Resolver,
    ShimOptions,
    TableGenOptions,
    UnresolvedPolicy,
)
from numbast.backends.tablegen.facts import header_facts
from numbast.backends.tablegen.function import (
    TableGenFunctionsRenderer,
    classify_params,
    mnemonic_for,
)
from numbast.backends.tablegen.lowering import LoweringRenderer
from numbast.backends.tablegen.scaffold import ScaffoldRenderer
from numbast.backends.tablegen.shim import (
    ShimRenderer,
    sanitize_c_name,
    shim_lookup_name,
)
from numbast.backends.tablegen.types import (
    TypeMapper,
    abi_c_type,
    llvm_token,
)
from numbast.backends.tablegen.types_td import (
    TableGenTypesRenderer,
    render_dialect_td,
)
from numbast.backends.tablegen.typedefs import recover_typedefs

__all__ = [
    "LoweringRenderer",
    "OutParamPolicy",
    "Resolver",
    "ScaffoldRenderer",
    "ShimOptions",
    "ShimRenderer",
    "TableGenConfig",
    "TableGenFunctionsRenderer",
    "TableGenOptions",
    "TableGenTypesRenderer",
    "TypeMapper",
    "UnresolvedPolicy",
    "abi_c_type",
    "classify_params",
    "header_facts",
    "llvm_token",
    "mnemonic_for",
    "recover_typedefs",
    "render_dialect_td",
    "sanitize_c_name",
    "shim_lookup_name",
]


def __getattr__(name):
    # TableGenConfig subclasses the core Config, which pulls in the Numba
    # stack; keep that out of the import path unless it is asked for.
    if name == "TableGenConfig":
        from numbast.backends.tablegen.config import TableGenConfig

        return TableGenConfig
    raise AttributeError(name)
