# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated C access layer."""

import pytest

from numbast.backends.tablegen.config import (
    Resolver,
    ShimOptions,
    TableGenOptions,
    UnresolvedPolicy,
)
from numbast.backends.tablegen.shim import (
    ShimRenderer,
    sanitize_c_name,
    shim_lookup_name,
)
from numbast.model import FunctionPlan


class _Type:
    def __init__(self, name):
        self.name = name
        self.unqualified_non_ref_type_name = name.replace("const ", "")


class _Param:
    def __init__(self, name):
        self.name = name


class _Decl:
    def __init__(self, name, ret, params):
        self.name = name
        self.return_type = _Type(ret)
        self.params = [_Param(p) for _, p in params]
        self.param_types = [_Type(t) for t, _ in params]


def _plan(name, ret="AcmeResult", params=()):
    return FunctionPlan(
        decl=_Decl(name, ret, params),
        exposed_name=name.removeprefix("acme"),
        use_cooperative=False,
    )


def _options(**kw):
    base = dict(
        dialect_name="acme",
        cpp_class_prefix="Acme",
        status_type="AcmeResult",
        shim=ShimOptions(
            soname="libacme.so.1",
            resolver=Resolver.DLSYM,
            unresolved_policy=UnresolvedPolicy.ABORT,
        ),
    )
    base.update(kw)
    return TableGenOptions(**base)


def _render(plans, options, **kw):
    # `AcmeResult` has to be declared an enum: `abi_c_type` refuses to guess a
    # width for a name it does not recognise rather than silently pick `int`,
    # so omitting it is a parse-vocabulary error, not a default.
    kw.setdefault("enums", ("AcmeResult",))
    return ShimRenderer(plans, options, **kw).render_as_str()


def test_no_shim_when_unconfigured():
    """With no ``Shim`` block there is no access layer to emit."""
    out = _render([_plan("acmeInit")], _options(shim=None))
    assert out == ""


def test_wrapper_shape():
    out = _render(
        [_plan("acmeStreamCreate", params=[("AcmeStream_st **", "phStream")])],
        _options(),
    )
    assert 'extern "C" int _acmeStreamCreate(void * phStream) {' in out
    assert '_shim_resolve<int (*)(void *)>("acmeStreamCreate")' in out
    assert "return _shim_fn(phStream);" in out


def test_void_parameter_list_is_spelled_void():
    """A C function with no parameters is ``(void)``, not ``()``."""
    out = _render([_plan("acmeShutdown")], _options())
    assert 'extern "C" int _acmeShutdown(void) {' in out
    assert "return _shim_fn();" in out


def test_header_is_never_included():
    """ABI mode: the shim must compile without the library's SDK."""
    out = _render(
        [_plan("acmeInit", params=[("unsigned int", "flags")])], _options()
    )
    assert "#include <dlfcn.h>" in out
    assert "acme.h" not in out
    assert "AcmeResult" not in out


def test_dlsym_resolver_looks_up_the_declared_symbol():
    """``dlsym`` can only find what is actually exported."""
    out = _render([_plan("acmeMemAlloc_v2")], _options())
    assert '"acmeMemAlloc_v2"' in out


def test_proc_address_resolver_looks_up_the_base_name():
    """The versioned resolver is what selects the variant, so ask for the base."""
    options = _options(
        shim=ShimOptions(
            soname="libacme.so.1",
            resolver=Resolver.GET_PROC_ADDRESS,
            proc_address_symbol="acmeGetProcAddress",
            version=12090,
        )
    )
    out = _render([_plan("acmeMemAlloc_v2")], options)
    # The wrapper keeps the real symbol; only the lookup is version-less.
    assert 'extern "C" int _acmeMemAlloc_v2(' in out
    assert '"acmeMemAlloc"' in out
    assert "12090" in out


def test_status_policy_degrades_in_band():
    """A status-returning entry point can report "not found" as an API error."""
    options = _options(
        shim=ShimOptions(
            soname="libacme.so.1",
            unresolved_policy=UnresolvedPolicy.STATUS,
            not_found_value=500,
        )
    )
    out = _render([_plan("acmeInit")], options)
    assert "if (!_shim_fn) return 500;" in out


def test_status_policy_cannot_apply_to_a_void_function():
    """There is no status channel to report through, so it must abort.

    The policy is a request, not a guarantee: honouring it here would mean
    emitting ``return 500;`` from a ``void`` function, which does not compile.
    """
    options = _options(
        shim=ShimOptions(
            soname="libacme.so.1",
            unresolved_policy=UnresolvedPolicy.STATUS,
            not_found_value=500,
        )
    )
    out = _render([_plan("acmeShutdown", ret="void")], options)
    assert 'extern "C" void _acmeShutdown(void)' in out
    assert "return 500" not in out
    assert "_shim_unresolved" in out


def test_unnamed_parameters_get_positional_names():
    """A header may leave a parameter unnamed; C still needs an identifier."""
    out = _render(
        [_plan("acmeInit", params=[("unsigned int", "")])], _options()
    )
    assert 'extern "C" int _acmeInit(unsigned int arg0) {' in out


def test_parameter_named_like_a_shim_local_is_renamed():
    """``_shim_fn`` is bound in the wrapper body, so a parameter cannot use it."""
    out = _render(
        [_plan("acmeSetHook", params=[("void *", "_shim_fn")])], _options()
    )
    assert "_shim_fn_" in out


@pytest.mark.parametrize(
    "name,expected",
    [("fn", "fn"), ("", "arg0"), ("int", "int_"), ("2nd", "_2nd")],
)
def test_sanitize_c_name(name, expected):
    assert sanitize_c_name(name, 0) == expected


def test_sanitize_c_name_does_not_rename_mlir_reserved_words():
    """``value`` and ``result`` are reserved for *TableGen*, not for C.

    Applying the TableGen reserved set here would rename C parameters for a
    reason that does not apply to C, and the shim would then disagree with a
    hand-written reference for no reason.
    """
    assert sanitize_c_name("value", 0) == "value"
    assert sanitize_c_name("result", 0) == "result"


@pytest.mark.parametrize(
    "symbol,base",
    [
        ("acmeMemAlloc_v2", "acmeMemAlloc"),
        ("acmeMemAlloc", "acmeMemAlloc"),
        ("acmeThing_v12", "acmeThing"),
    ],
)
def test_shim_lookup_name(symbol, base):
    assert shim_lookup_name(symbol, True) == base
    assert shim_lookup_name(symbol, False) == symbol
