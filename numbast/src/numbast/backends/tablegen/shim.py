# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the C access layer the generated ops lower to.

Ops lower to ``_<symbol>`` rather than to ``<symbol>`` directly, and this
renders the ``_<symbol>`` layer. Three reasons it exists, in decreasing order of
how often they matter:

* **There is a symbol to call.** A lowering has to name something the linker
  or loader can find. Going through a generated wrapper makes "who provides
  this" answerable by the library team rather than by the compiler.
* **Version selection happens in one place.** With a resolver-based lookup the
  shim asks the library for the entry point appropriate to the version it was
  generated against, so one dialect build spans library versions.
* **ABI decoupling.** The dialect and any IR written against it are unaffected
  when the library changes which variant a name resolves to.

The shim is emitted in *ABI mode*: it never includes the library's headers.
Every type is reduced to a header-free equivalent by
:func:`~numbast.backends.tablegen.types.abi_c_type`, so the generated file
compiles against nothing but ``dlfcn.h``. That is what makes it buildable in an
environment that has the library's ``.so`` but not its SDK.
"""

import re

from numbast.backends.tablegen.config import (
    Resolver,
    TableGenOptions,
    UnresolvedPolicy,
)
from numbast.backends.tablegen.types import abi_c_type

__all__ = ["ShimRenderer", "sanitize_c_name", "shim_lookup_name"]

# C keywords plus the two identifiers the generated wrapper body binds. A
# parameter sharing either name would shadow it -- cuda.h really does declare
# `CUhostFn fn`, which is why the locals are `_shim_`-prefixed rather than
# something natural like `fn`.
#
# Deliberately NOT the TableGen `_RESERVED` set from `function.py`: that one
# exists because mlir-tblgen generates accessors with those names, which is a
# fact about MLIR and has nothing to do with C. Applying it here renames C
# parameters for a reason that does not apply to C, and the shim then disagrees
# with a hand-written reference for no reason.
_C_RESERVED = frozenset(
    {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        # bound in the generated wrapper body
        "_shim_fn",
        "_shim_resolve",
    }
)


def sanitize_c_name(name: str, index: int = 0, taken=()) -> str:
    """Make ``name`` usable as a C parameter name, and unique.

    Headers may leave a parameter unnamed, in which case there is nothing to
    carry over and a positional name is synthesised.
    """
    out = name or f"arg{index}"
    out = re.sub(r"[^A-Za-z0-9_]", "_", out)
    if out[:1].isdigit():
        out = f"_{out}"
    while out in _C_RESERVED or out in taken:
        out += "_"
    return out


def shim_lookup_name(c_symbol: str, resolve_base_name: bool) -> str:
    """The name to ask the library for.

    With a versioned resolver the *base* name is the right thing to look up --
    the resolver is what picks the variant, so asking for ``cuMemAlloc_v2``
    bypasses the mechanism that makes the shim version-portable. With a plain
    ``dlsym`` the opposite holds: only the declared symbol is exported.
    """
    if not resolve_base_name:
        return c_symbol
    return re.sub(r"_v\d+$", "", c_symbol)


_PROC_ADDRESS_LOADER = """\
using _proc_fn = int (*)(const char *, void **, int, unsigned long long, void *);

void *g_handle = nullptr;
_proc_fn g_getproc = nullptr;

bool ensure_loaded() {{
  if (g_getproc) return true;
  if (!g_handle) g_handle = dlopen("{soname}", RTLD_NOW | RTLD_GLOBAL);
  if (!g_handle) return false;
  g_getproc = reinterpret_cast<_proc_fn>(dlsym(g_handle, "{proc}"));
  return g_getproc != nullptr;
}}

// Resolve `name` for the compiled version, falling back to a raw dlsym.
template <typename F> F _shim_resolve(const char *name) {{
  if (!ensure_loaded()) return nullptr;
  void *fptr = nullptr;
  int r = g_getproc(name, &fptr, {version}, 0, nullptr);
  if (r != {success} || !fptr) fptr = dlsym(g_handle, name);
  return reinterpret_cast<F>(fptr);
}}"""

_DLSYM_LOADER = """\
void *g_handle = nullptr;

bool ensure_loaded() {{
  if (g_handle) return true;
  g_handle = dlopen("{soname}", RTLD_NOW | RTLD_GLOBAL);
  return g_handle != nullptr;
}}

template <typename F> F _shim_resolve(const char *name) {{
  if (!ensure_loaded()) return nullptr;
  return reinterpret_cast<F>(dlsym(g_handle, name));
}}"""

_FILE = """\
// Auto-generated. Do not edit by hand.
//
// Access layer for the {dialect} dialect -- ABI mode (no library header),
// resolver "{resolver}". Types are reduced to their ABI form; each `_<symbol>`
// resolves its entry point at runtime and forwards through it.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

{aggregates}

namespace {{

{loader}

// Raised when an entry point can't be resolved in the loaded library -- a
// linkage failure, distinct from an API error.
// `unused` keeps it warning-free for recipes that never take this path.
__attribute__((noreturn, unused)) void _shim_unresolved(const char *name) {{
  fprintf(stderr, "{dialect}: unresolved entry point '%s' in %s\\n", name, "{soname}");
  abort();
}}

}}  // namespace

{functions}
"""

_FUNCTION = """\
extern "C" {ret} {shim_symbol}({params}) {{
  static auto _shim_fn = _shim_resolve<{ret} (*)({fp_types})>("{lookup}");{guard}
  return _shim_fn({args});
}}"""


class ShimRenderer:
    """Render the access layer for a set of function plans.

    Takes the same inputs as
    :class:`~numbast.backends.tablegen.function.TableGenFunctionsRenderer`, so
    the two cannot disagree about which entry points exist or what their
    parameters are.
    """

    def __init__(
        self,
        plans,
        config: TableGenOptions,
        *,
        records: dict | None = None,
        enums=(),
        typedefs=None,
    ):
        self._plans = list(plans)
        self._cfg = config
        self._records = records or {}
        self._enums = enums or ()
        self._typedefs = typedefs
        # name -> size, filled in while rendering; the caller emits the PODs.
        self._aggregates: dict[str, int] = {}

    @property
    def by_value_aggregates(self) -> dict:
        """By-value aggregate parameters seen, as ``name -> byte size``."""
        return dict(self._aggregates)

    def _abi(self, spelling: str) -> str:
        return abi_c_type(
            spelling,
            enums=self._enums,
            records=self._records,
            by_value_aggregates=self._aggregates,
        )

    def _returns_status(self, decl) -> bool:
        """Whether this entry point reports through the status channel.

        Only then can an unresolved entry point be reported in band; a void or
        value-returning function has nowhere to put it.
        """
        cfg = self._cfg
        if not cfg.status_type:
            return False
        spelling = decl.return_type.unqualified_non_ref_type_name.strip()
        public = (
            self._typedefs.public(spelling)
            if self._typedefs is not None
            else spelling
        )
        return public == cfg.status_type

    def _render_one(self, plan) -> str:
        decl = plan.decl
        shim = self._cfg.shim

        ret = self._abi(decl.return_type.unqualified_non_ref_type_name.strip())

        taken: set[str] = set()
        names, types = [], []
        for i, (param, ptype) in enumerate(zip(decl.params, decl.param_types)):
            nm = sanitize_c_name(param.name, i, taken)
            taken.add(nm)
            names.append(nm)
            types.append(self._abi(ptype.unqualified_non_ref_type_name))

        if names:
            params = ", ".join(f"{t} {n}" for t, n in zip(types, names))
            fp_types = ", ".join(types)
            args = ", ".join(names)
        else:
            params, fp_types, args = "void", "void", ""

        lookup = shim_lookup_name(plan.c_symbol, shim.resolve_base_name)

        if (
            shim.unresolved_policy == UnresolvedPolicy.STATUS
            and self._returns_status(decl)
        ):
            guard = f"\n  if (!_shim_fn) return {shim.not_found_value};"
        else:
            guard = f'\n  if (!_shim_fn) _shim_unresolved("{lookup}");'

        return _FUNCTION.format(
            ret=ret,
            shim_symbol=f"{self._cfg.shim_prefix}{plan.c_symbol}",
            params=params,
            fp_types=fp_types,
            lookup=lookup,
            guard=guard,
            args=args,
        )

    def _render_aggregates(self) -> str:
        """By-value aggregates as opaque, correctly-sized PODs.

        Size-only. Correct when the aggregate is passed in memory or is small
        and all-integer; a small struct with float fields, an over-aligned
        struct, or a struct return needs real target-specific calling
        convention classification, which is the single seam where that would go.
        The CUDA driver passes every aggregate by pointer, so this stays empty
        for it.
        """
        if not self._aggregates:
            return "// (no by-value aggregate params)"
        return "\n".join(
            f"typedef struct {{ char _[{size}]; }} {name};"
            for name, size in sorted(self._aggregates.items())
        )

    def render_as_str(self) -> str:
        """The complete shim translation unit, or ``""`` if none was asked for."""
        cfg = self._cfg
        shim = cfg.shim
        if shim is None:
            return ""

        # Render functions first: this is what populates `_aggregates`, which
        # the preamble above them declares.
        functions = "\n\n".join(self._render_one(p) for p in self._plans)

        if shim.resolver == Resolver.GET_PROC_ADDRESS:
            loader = _PROC_ADDRESS_LOADER.format(
                soname=shim.soname,
                proc=shim.proc_address_symbol,
                version=shim.version or 0,
                success=cfg.status_success,
            )
        else:
            loader = _DLSYM_LOADER.format(soname=shim.soname)

        return _FILE.format(
            dialect=cfg.dialect_name,
            resolver=shim.resolver,
            soname=shim.soname,
            aggregates=self._render_aggregates(),
            loader=loader,
            functions=functions,
        )
