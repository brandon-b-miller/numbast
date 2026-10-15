# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the C++ and CMake around the generated ``.td`` files.

A dialect is not a set of TableGen definitions; it is a shared object that can
be loaded into an MLIR context. Between the two sit a handful of files that are
pure boilerplate but have to be exactly right: the headers that include what
mlir-tblgen's output references, the ``initialize()`` that registers the types
and ops, a C entry point so the dialect can be loaded by ``ctypes``, and the
CMake that runs mlir-tblgen and links the result.

None of it depends on the library being bound -- every file here is a function
of the dialect's name and prefix alone -- which is exactly why it is worth
generating rather than copying by hand into each new dialect.

The link model
--------------
The library this CMake declares deliberately leaves MLIR's core symbols
**undefined**, to be resolved at ``dlopen`` time against the MLIR already loaded
in the host process. Linking MLIR in here would create a second copy, and
because MLIR identifies a type by the address of a per-type static, two copies
are two disjoint type universes: a dialect registered through one is absent from
the other, ``isa<>`` across the boundary is false, and nothing crashes.
"""

__all__ = ["ScaffoldRenderer"]


class ScaffoldRenderer:
    """Render the non-TableGen files of a generated dialect.

    Parameters
    ----------
    config:
        :class:`~numbast.backends.tablegen.config.TableGenOptions`.
    has_types:
        Whether any ``TypeDef`` will be emitted. Always true in practice -- the
        opaque pointer type is unconditional -- but the generated
        ``initialize()`` must not reference a ``Types.cpp.inc`` that was never
        produced.
    has_ops:
        Whether any op will be emitted.
    has_shim:
        Whether a shim translation unit will be emitted, which decides if the
        runtime subdirectory exists.
    """

    def __init__(
        self,
        config,
        *,
        has_types: bool = True,
        has_ops: bool = True,
        has_shim: bool = False,
    ):
        self._cfg = config
        self._has_types = has_types
        self._has_ops = has_ops
        self._has_shim = has_shim

    # -- naming -----------------------------------------------------------

    @property
    def include_root(self) -> str:
        """Top include directory, e.g. ``cudadrv_dialect``.

        Matches what the generated ``.td`` files already ``include``, so the two
        cannot drift.
        """
        return f"{self._cfg.dialect_name}_dialect"

    @property
    def _ns(self) -> str:
        """C++ namespace the generated classes land in.

        Derived from the dialect name exactly as ``render_dialect_td`` derives
        ``cppNamespace``, rather than read from config: the ``using namespace``
        in these files has to name what TableGen actually emitted, and two
        independent spellings of it would compile right up until they differed.
        """
        return f"mlir::{self._cfg.dialect_name}"

    @property
    def _ir_dir(self) -> str:
        return f"{self.include_root}/Dialect/{self._cfg.cpp_class_prefix}/IR"

    # -- dialect ----------------------------------------------------------

    def dialect_h(self) -> str:
        prefix = self._cfg.cpp_class_prefix
        guard = f"{prefix.upper()}_DIALECT_H"
        return f"""#ifndef {guard}
#define {guard}

#include "mlir/IR/Dialect.h"

#include "{self._ir_dir}/{prefix}Dialect.h.inc"

#endif // {guard}
"""

    def dialect_cpp(self) -> str:
        """``initialize()`` registering whatever was emitted.

        The ``GET_TYPEDEF_LIST`` / ``GET_OP_LIST`` include-as-list idiom is how
        MLIR expects a dialect to enumerate its own contents.
        """
        prefix = self._cfg.cpp_class_prefix
        includes, blocks = [], []

        if self._has_types:
            includes.append(f'#include "{self._ir_dir}/{prefix}Types.h"')
            blocks.append(
                "  addTypes<\n"
                "#define GET_TYPEDEF_LIST\n"
                f'#include "{self._ir_dir}/{prefix}Types.cpp.inc"\n'
                "      >();"
            )
        if self._has_ops:
            includes.append(f'#include "{self._ir_dir}/{prefix}Ops.h"')
            blocks.append(
                "  addOperations<\n"
                "#define GET_OP_LIST\n"
                f'#include "{self._ir_dir}/{prefix}Ops.cpp.inc"\n'
                "      >();"
            )

        includes_block = "\n".join(includes)
        if includes_block:
            includes_block += "\n"
        body = "\n".join(blocks) or "  // Nothing registered yet."

        return f"""#include "{self._ir_dir}/{prefix}Dialect.h"
{includes_block}
using namespace mlir;
using namespace {self._ns};

#include "{self._ir_dir}/{prefix}Dialect.cpp.inc"

void {prefix}Dialect::initialize() {{
{body}
}}
"""

    # -- types ------------------------------------------------------------

    def types_h(self) -> str:
        prefix = self._cfg.cpp_class_prefix
        guard = f"{prefix.upper()}_TYPES_H"
        return f"""#ifndef {guard}
#define {guard}

#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "{self._ir_dir}/{prefix}Types.h.inc"

#endif // {guard}
"""

    def types_cpp(self) -> str:
        """Includes matter here.

        Because the dialect sets ``useDefaultTypePrinterParser``, mlir-tblgen
        also emits the dispatching ``parseType``/``printType`` into this
        ``.cpp.inc``. Those reference ``mlir::OptionalParseResult``,
        ``mlir::AsmPrinter`` and ``mlir::DialectAsmParser``, which live in
        ``DialectImplementation.h``. Without it the build fails on incomplete
        types inside generated code, which reads like a generator bug.
        """
        prefix = self._cfg.cpp_class_prefix
        return f"""#include "{self._ir_dir}/{prefix}Types.h"

#include "{self._ir_dir}/{prefix}Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace {self._ns};

#define GET_TYPEDEF_CLASSES
#include "{self._ir_dir}/{prefix}Types.cpp.inc"
"""

    # -- ops ----------------------------------------------------------------

    def ops_h(self) -> str:
        """MLIR auto-attaches interfaces (bytecode, asm) to every op, and the
        generated ``.h.inc`` references them, so their headers must be in scope.
        """
        prefix = self._cfg.cpp_class_prefix
        guard = f"{prefix.upper()}_OPS_H"
        types_include = (
            f'#include "{self._ir_dir}/{prefix}Types.h"\n'
            if self._has_types
            else ""
        )
        return f"""#ifndef {guard}
#define {guard}

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "{self._ir_dir}/{prefix}Dialect.h"
{types_include}
#define GET_OP_CLASSES
#include "{self._ir_dir}/{prefix}Ops.h.inc"

#endif // {guard}
"""

    def ops_cpp(self) -> str:
        prefix = self._cfg.cpp_class_prefix
        return f"""#include "{self._ir_dir}/{prefix}Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace {self._ns};

#define GET_OP_CLASSES
#include "{self._ir_dir}/{prefix}Ops.cpp.inc"
"""

    # -- CAPI registration --------------------------------------------------

    def capi_registration_cpp(self) -> str:
        """A C entry point that registers and loads the dialect in one call.

        MLIR's own macro gives ``mlirGetDialectHandle__<ns>__()``; this adds
        ``<name>RegisterAndLoad(MlirContext)`` on top so a caller needs one
        ``ctypes`` call and no extension module. ``MlirContext`` is a
        single-pointer struct, so under the SysV C ABI it is passed exactly like
        a ``void *`` -- which is what lets Python hand over the raw pointer it
        pulls out of the bindings' context capsule.
        """
        prefix = self._cfg.cpp_class_prefix
        name = self._cfg.dialect_name
        return f"""// Auto-generated. Do not edit by hand.
//
// C-API registration for the {name} dialect. Build this into a SHARED library
// with the MLIR core symbols left undefined (resolved at dlopen time from the
// host's already-loaded MLIR), then from Python:
//
//     lib = ctypes.CDLL("lib{prefix}Dialect.so", mode=ctypes.RTLD_GLOBAL)
//     lib.{name}RegisterAndLoad(raw_mlir_context_ptr)
//
// where raw_mlir_context_ptr is the void* inside the bindings' Context capsule.

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"

#include "{self._ir_dir}/{prefix}Dialect.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION({prefix}, {name}, {self._ns}::{prefix}Dialect)

extern "C" void {name}RegisterAndLoad(MlirContext context) {{
  MlirDialectHandle handle = mlirGetDialectHandle__{name}__();
  mlirDialectHandleRegisterDialect(handle, context);
  mlirDialectHandleLoadDialect(handle, context);
}}
"""

    # -- CMake ---------------------------------------------------------------

    def top_cmakelists(self) -> str:
        prefix = self._cfg.cpp_class_prefix
        runtime = (
            f"add_subdirectory(lib/Runtime/{prefix})\n"
            if self._has_shim
            else ""
        )
        return f"""# Auto-generated. Do not edit by hand.
cmake_minimum_required(VERSION 3.20)
project({self.include_root}_project LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${{MLIR_DIR}}")
message(STATUS "Using LLVMConfig.cmake in: ${{LLVM_DIR}}")

list(APPEND CMAKE_MODULE_PATH "${{MLIR_CMAKE_DIR}}")
list(APPEND CMAKE_MODULE_PATH "${{LLVM_CMAKE_DIR}}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${{LLVM_INCLUDE_DIRS}})
include_directories(${{MLIR_INCLUDE_DIRS}})
include_directories(${{CMAKE_CURRENT_SOURCE_DIR}}/include)
include_directories(${{CMAKE_CURRENT_BINARY_DIR}}/include)

add_subdirectory(include/{self.include_root}/Dialect/{prefix}/IR)
add_subdirectory(lib/Dialect/{prefix}/IR)
{runtime}"""

    def include_cmakelists(self) -> str:
        """Runs mlir-tblgen over each ``.td``.

        ``add_public_tablegen_target`` is what makes the generated ``.inc`` a
        real dependency; without it the library can start compiling before
        tablegen has run.
        """
        prefix = self._cfg.cpp_class_prefix
        name = self._cfg.dialect_name
        lines = [
            f"set(LLVM_TARGET_DEFINITIONS {prefix}Dialect.td)",
            f"mlir_tablegen({prefix}Dialect.h.inc -gen-dialect-decls -dialect={name})",
            f"mlir_tablegen({prefix}Dialect.cpp.inc -gen-dialect-defs -dialect={name})",
            f"add_public_tablegen_target(MLIR{prefix}DialectIncGen)",
        ]
        if self._has_types:
            lines += [
                "",
                f"set(LLVM_TARGET_DEFINITIONS {prefix}Types.td)",
                f"mlir_tablegen({prefix}Types.h.inc -gen-typedef-decls -typedefs-dialect={name})",
                f"mlir_tablegen({prefix}Types.cpp.inc -gen-typedef-defs -typedefs-dialect={name})",
                f"add_public_tablegen_target(MLIR{prefix}TypesIncGen)",
            ]
        if self._has_ops:
            lines += [
                "",
                f"set(LLVM_TARGET_DEFINITIONS {prefix}Ops.td)",
                f"mlir_tablegen({prefix}Ops.h.inc -gen-op-decls)",
                f"mlir_tablegen({prefix}Ops.cpp.inc -gen-op-defs)",
                f"add_public_tablegen_target(MLIR{prefix}OpsIncGen)",
            ]
        return "\n".join(lines) + "\n"

    def lib_cmakelists(self) -> str:
        prefix = self._cfg.cpp_class_prefix
        sources = [f"{prefix}Dialect.cpp"]
        depends = [f"MLIR{prefix}DialectIncGen"]
        if self._has_types:
            sources.append(f"{prefix}Types.cpp")
            depends.append(f"MLIR{prefix}TypesIncGen")
        if self._has_ops:
            sources.append(f"{prefix}Ops.cpp")
            depends.append(f"MLIR{prefix}OpsIncGen")

        return f"""add_mlir_library(MLIR{prefix}Dialect
  {chr(10).join(sources).replace(chr(10), chr(10) + "  ")}

  ADDITIONAL_HEADER_DIRS
  ${{CMAKE_SOURCE_DIR}}/include/{self.include_root}/Dialect/{prefix}/IR

  DEPENDS
  {chr(10).join(depends).replace(chr(10), chr(10) + "  ")}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
"""

    def runtime_cmakelists(self) -> str:
        """Builds the shim SHARED so the ExecutionEngine can ``dlopen`` it.

        Default visibility because the JIT resolves ``_<symbol>`` through the
        process symbol table; hiding them would make the JIT fail to find the
        very functions this exists to provide.
        """
        prefix = self._cfg.cpp_class_prefix
        return f"""# Auto-generated. Do not edit by hand.

add_library({prefix}Shim SHARED {prefix}Shim.cpp)
set_target_properties({prefix}Shim PROPERTIES
  CXX_VISIBILITY_PRESET default
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories({prefix}Shim PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}})
target_link_libraries({prefix}Shim PUBLIC ${{CMAKE_DL_LIBS}})
"""

    def gitignore(self) -> str:
        return "build/\n"
