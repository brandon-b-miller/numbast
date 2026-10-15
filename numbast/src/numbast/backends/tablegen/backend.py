# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the TableGen backend: header in, buildable dialect tree out.

This is the piece that makes the backend selectable by name. Everything it does
is sequencing -- parse, curate, render, write -- and each stage is a module that
can be tested without it.

Why the parse defaults differ from the Numba backends
-----------------------------------------------------
Two of numbast's defaults are wrong for this backend and silently produce an
empty dialect rather than an error:

``cuda_header_mode``
    Off by default. Left off, a CUDA *driver* header parses to zero
    declarations, because clang's implicit CUDA include injection attributes
    them elsewhere.

``skip_non_device``
    On by default, which is right for the device headers the Numba backends
    bind and exactly wrong here: a host C API has no ``__device__`` functions at
    all, so every entry point is curated away.

Both are therefore configurable, and both default the other way in
:class:`~numbast.backends.tablegen.config.TableGenOptions`.
"""

import os
import re
from dataclasses import replace

import click
import yaml

from ast_canopy import parse_declarations_from_source

from numbast.backends.tablegen.config import TableGenConfig
from numbast.backends.tablegen.facts import header_facts
from numbast.backends.tablegen.function import TableGenFunctionsRenderer
from numbast.backends.tablegen.lowering import LoweringRenderer
from numbast.backends.tablegen.scaffold import ScaffoldRenderer
from numbast.backends.tablegen.shim import ShimRenderer
from numbast.backends.tablegen.types_td import (
    TableGenTypesRenderer,
    render_dialect_td,
)
from numbast.curate import plan_functions
from numbast.tools.yaml_tags import string_constructor

__all__ = ["generate_dialect", "static_binding_generator", "strip_versions"]

# Stream-variant and version suffixes, longest first so `_v2_ptds` is fully
# removed in one pass.
_VERSION_SUFFIX = re.compile(r"(_v\d+)?(_ptds|_ptsz)?$")


class _ConfigLoader(yaml.SafeLoader):
    pass


_ConfigLoader.add_constructor("!numbast_join", string_constructor)


def _base_name(name: str) -> str:
    return _VERSION_SUFFIX.sub("", name, count=1)


def strip_versions(plans):
    """Collapse versioned entry points onto their base name.

    Only the *exposed* name is rewritten; ``c_symbol`` keeps the real exported
    symbol, because that is what has to be linked or resolved. The first
    declaration of a base name wins, which is header order, and that is the one
    the library's own header makes current.

    Without this a dialect carries ``mem.alloc.v2`` alongside ``mem.alloc`` and
    several near-duplicate ops whose only difference is which ABI generation
    they came from.
    """
    seen: set[str] = set()
    out = []
    for plan in plans:
        base = _base_name(plan.exposed_name)
        if base in seen:
            continue
        seen.add(base)
        out.append(
            plan
            if base == plan.exposed_name
            else replace(plan, exposed_name=base)
        )
    return out


def generate_dialect(config, output_dir: str, *, bypass_parse_error=False):
    """Render the dialect tree described by ``config`` into ``output_dir``.

    Returns the mapping of relative path -> contents that was written, so a
    caller (or a test) can assert on the tree without reading it back.
    """
    options = config.tablegen

    entry_point = os.path.abspath(config.entry_point)
    retain_list = [os.path.abspath(p) for p in config.retain_list]
    if not config.gpu_arch:
        raise ValueError("At least one GPU architecture must be provided.")

    decls = parse_declarations_from_source(
        entry_point,
        retain_list,
        compute_capability=config.gpu_arch[0],
        additional_includes=config.clang_includes_paths,
        defines=config.predefined_macros,
        bypass_parse_error=bypass_parse_error,
        cuda_header_mode=options.cuda_header_mode,
    )

    facts = header_facts(decls, entry_point)

    plans = plan_functions(
        decls.functions,
        header_path=entry_point,
        excludes=config.exclude_functions,
        skip_prefix=config.skip_prefix,
        skip_non_device=options.skip_non_device,
        cooperative_launch_required=(
            config.cooperative_launch_required_functions_regex
        ),
        prefix_removal=config.api_prefix_removal.get("Function", []),
        argument_intents=config.function_argument_intents,
    )
    if options.strip_version_suffixes:
        plans = strip_versions(plans)
    if options.symbols:
        # Matched against the version-less C symbol, not the exposed name: the
        # list states which entry points of the library are covered, which is a
        # fact about the library and should not move when prefix removal or the
        # dialect's naming changes.
        order = {name: i for i, name in enumerate(options.symbols)}
        plans = [p for p in plans if _base_name(p.c_symbol) in order]
        # Emit in the order the coverage list states, not header order. With an
        # explicit list the list *is* the author's ordering, and it is stable
        # across header revisions in a way declaration order is not -- a header
        # that moves a declaration would otherwise reorder the whole dialect.
        plans.sort(key=lambda p: order[_base_name(p.c_symbol)])

    # Ops first: rendering is what populates the mapper's type registry, which
    # the types renderer and the lowering descriptor both read.
    ops = TableGenFunctionsRenderer(
        plans,
        options,
        records=facts["records"],
        enums=facts["enums"],
        typedefs=facts["typedefs"],
        record_objs=facts["record_objs"],
    )
    ops_td = ops.render_as_str()

    types = TableGenTypesRenderer(ops.mapper, options, unions=facts["unions"])
    types_td = types.render_as_str()

    shim = ShimRenderer(
        plans,
        options,
        records=facts["record_objs"],
        enums=facts["enums"],
        typedefs=facts["typedefs"],
    )
    shim_cpp = shim.render_as_str()

    lowering = LoweringRenderer(
        plans,
        options,
        mapper=ops.mapper,
        types_renderer=types,
        records=facts["record_objs"],
        enums=facts["enums"],
        typedefs=facts["typedefs"],
    )

    scaffold = ScaffoldRenderer(
        options,
        has_types=True,
        has_ops=bool(plans),
        has_shim=bool(shim_cpp),
    )

    prefix = options.cpp_class_prefix
    ir = f"include/{scaffold.include_root}/Dialect/{prefix}/IR"
    lib = f"lib/Dialect/{prefix}/IR"

    tree = {
        "CMakeLists.txt": scaffold.top_cmakelists(),
        ".gitignore": scaffold.gitignore(),
        f"{ir}/{prefix}Dialect.td": render_dialect_td(options),
        f"{ir}/{prefix}Dialect.h": scaffold.dialect_h(),
        f"{ir}/{prefix}Types.td": types_td,
        f"{ir}/{prefix}Types.h": scaffold.types_h(),
        f"{ir}/{prefix}Ops.td": ops_td,
        f"{ir}/{prefix}Ops.h": scaffold.ops_h(),
        f"{ir}/CMakeLists.txt": scaffold.include_cmakelists(),
        f"{lib}/{prefix}Dialect.cpp": scaffold.dialect_cpp(),
        f"{lib}/{prefix}Types.cpp": scaffold.types_cpp(),
        f"{lib}/{prefix}Ops.cpp": scaffold.ops_cpp(),
        f"{lib}/CMakeLists.txt": scaffold.lib_cmakelists(),
        f"lib/CAPI/{prefix}Registration.cpp": scaffold.capi_registration_cpp(),
    }

    if options.runtime_package:
        tree[f"{options.dialect_name}_lowering.py"] = lowering.render_as_str()

    if shim_cpp:
        tree[f"lib/Runtime/{prefix}/{prefix}Shim.cpp"] = shim_cpp
        tree[f"lib/Runtime/{prefix}/CMakeLists.txt"] = (
            scaffold.runtime_cmakelists()
        )

    for relative, contents in tree.items():
        path = os.path.join(output_dir, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(contents)

    return tree


@click.command()
@click.pass_context
@click.option(
    "--cfg-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    required=True,
)
@click.option("-fmt", "--run-ruff-format", type=bool, default=True)
@click.option("-noraise", "--bypass-parse-error", type=bool, default=False)
@click.option("--backend", type=str, default=None)
def static_binding_generator(
    ctx,
    cfg_path,
    output_dir,
    run_ruff_format,
    bypass_parse_error,
    backend,
):
    """Emit an MLIR dialect for the API described by the config."""
    with open(cfg_path) as config_file:
        config_dict = yaml.load(config_file, Loader=_ConfigLoader)

    config = TableGenConfig(config_dict)
    tree = generate_dialect(
        config, output_dir, bypass_parse_error=bypass_parse_error
    )
    click.echo(
        f"{config.tablegen.dialect_name}: wrote {len(tree)} files to "
        f"{output_dir}"
    )
