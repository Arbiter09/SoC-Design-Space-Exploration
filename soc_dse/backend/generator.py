"""
RTL Configuration Generator.

Reads a parameter space from a YAML file, enumerates all combinations
(or a random sample), renders Jinja2 Verilog templates into per-config
output directories, and registers each config in the SQLite database.

Environment variables
---------------------
SOC_TEMPLATES_DIR   Path to the Jinja2 template directory
                    (default: soc_dse/templates/)
SOC_CONFIGS_DIR     Root output directory for generated configs
                    (default: soc_dse/configs/)
SOC_DB_PATH         SQLite database path (default: soc_dse/dse.db)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterator

import math

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from soc_dse.backend import db

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR: Path = Path(os.environ.get("SOC_TEMPLATES_DIR", _ROOT / "templates"))
CONFIGS_DIR: Path = Path(os.environ.get("SOC_CONFIGS_DIR", _ROOT / "configs"))

# Verilog template filenames (without .j2 suffix → output filename)
_TEMPLATES: list[tuple[str, str]] = [
    ("pipeline.v.j2", "pipeline.v"),
    ("cache.v.j2", "cache.v"),
    ("soc_top.v.j2", "soc_top.v"),
]


# ---------------------------------------------------------------------------
# Config ID
# ---------------------------------------------------------------------------

def make_config_id(params: dict[str, Any]) -> str:
    """Return an 8-character deterministic hex ID for a parameter dict."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Parameter space enumeration
# ---------------------------------------------------------------------------

def _load_param_space(yaml_path: Path) -> dict[str, list[Any]]:
    """Load and validate a parameter space YAML file."""
    with yaml_path.open() as fh:
        space: dict[str, Any] = yaml.safe_load(fh)
    if not isinstance(space, dict):
        raise ValueError(f"Parameter space YAML must be a mapping; got {type(space)}")
    for key, val in space.items():
        if not isinstance(val, list) or len(val) == 0:
            raise ValueError(
                f"Parameter '{key}' must be a non-empty list of values, got: {val!r}"
            )
    return space  # type: ignore[return-value]


def enumerate_configs(
    param_space: dict[str, list[Any]],
    *,
    sample: int | None = None,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    Yield parameter dicts for each point in the Cartesian product.

    Parameters
    ----------
    param_space:
        Mapping from parameter name to list of candidate values.
    sample:
        If given, yield a random subset of this many configs instead of
        the full grid.
    seed:
        Random seed used when *sample* is set.
    """
    keys = list(param_space.keys())
    all_combos = list(itertools.product(*(param_space[k] for k in keys)))

    if sample is not None and sample < len(all_combos):
        rng = random.Random(seed)
        all_combos = rng.sample(all_combos, sample)

    for values in all_combos:
        yield dict(zip(keys, values))


# ---------------------------------------------------------------------------
# Jinja2 rendering
# ---------------------------------------------------------------------------

def _build_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def _derive_cache_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Compute Verilog-level cache geometry from high-level params.

    These derived values are injected into the Jinja2 render context so
    templates never need to perform log2 arithmetic themselves.
    """
    cache_size_kb: int = params["cache_size_kb"]
    memory_banks: int = params["memory_banks"]
    bus_width: int = params["bus_width"]

    bytes_per_word = bus_width // 8
    total_words = (cache_size_kb * 1024) // bytes_per_word
    words_per_bank = max(total_words // memory_banks, 1)

    # Bit-widths (ceil-log2, minimum 1)
    addr_bits = max(int(math.ceil(math.log2(total_words))), 1)
    bank_sel_bits = max(int(math.ceil(math.log2(memory_banks))), 1)
    index_bits = max(int(math.ceil(math.log2(words_per_bank))), 1)
    tag_bits = max(addr_bits - index_bits, 1)

    return {
        "addr_bits": addr_bits,
        "index_bits": index_bits,
        "tag_bits": tag_bits,
        "words_per_bank": words_per_bank,
        "bank_sel_bits": bank_sel_bits,
    }


def render_config(params: dict[str, Any], config_id: str) -> Path:
    """
    Render all Verilog templates for *params* into
    ``CONFIGS_DIR/<config_id>/rtl/`` and write ``params.yaml``.

    Returns the config directory path.
    """
    config_dir = CONFIGS_DIR / config_id
    rtl_dir = config_dir / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    # Write params.yaml alongside the RTL
    params_file = config_dir / "params.yaml"
    with params_file.open("w") as fh:
        yaml.safe_dump(params, fh, default_flow_style=False)

    # Build full render context: user params + derived cache geometry
    context = {**params, **_derive_cache_params(params)}

    env = _build_jinja_env()
    for template_name, output_name in _TEMPLATES:
        tmpl = env.get_template(template_name)
        rendered = tmpl.render(**context)
        (rtl_dir / output_name).write_text(rendered)
        log.debug("Rendered %s → %s", template_name, rtl_dir / output_name)

    return config_dir


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate_all(
    param_space_yaml: Path,
    *,
    sample: int | None = None,
    seed: int = 42,
    skip_existing: bool = True,
) -> list[str]:
    """
    Generate RTL for every config in the parameter space.

    Parameters
    ----------
    param_space_yaml:
        Path to the YAML file describing the parameter space.
    sample:
        Optional cap on the number of configs to generate.
    seed:
        RNG seed for sampling.
    skip_existing:
        When True, configs already in the SQLite DB are skipped.

    Returns
    -------
    list[str]
        Config IDs that were newly generated.
    """
    db.init_db()
    space = _load_param_space(param_space_yaml)
    generated: list[str] = []

    for params in enumerate_configs(space, sample=sample, seed=seed):
        config_id = make_config_id(params)

        if skip_existing and db.config_exists(config_id):
            log.debug("Skipping existing config %s", config_id)
            continue

        log.info("Generating config %s  params=%s", config_id, params)
        render_config(params, config_id)
        db.insert_config(config_id, params)
        generated.append(config_id)

    log.info("Generated %d new config(s)", len(generated))
    return generated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate RTL configs from a parameter space.")
    parser.add_argument(
        "param_space",
        type=Path,
        nargs="?",
        default=CONFIGS_DIR / "param_space.yaml",
        help="Path to param_space.yaml (default: soc_dse/configs/param_space.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit to N randomly sampled configs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-generate configs that already exist in the DB",
    )
    args = parser.parse_args()

    ids = generate_all(
        args.param_space,
        sample=args.sample,
        seed=args.seed,
        skip_existing=not args.no_skip,
    )
    print(f"Generated {len(ids)} config(s): {ids}")


if __name__ == "__main__":
    main()
