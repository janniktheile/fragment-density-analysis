#!/usr/bin/env python3
"""Fragment-resolved integration of Gaussian Cube volumetric data.

The script partitions a scalar field among user-defined molecular fragments using
smooth distance-based weights. For each grid point r and fragment F,

    s_F(r) = exp[-d_F(r)^2 / (2 sigma^2)]
    w_F(r) = s_F(r) / sum_G s_G(r)

where d_F(r) is the distance to the nearest atom in fragment F. The integrated
fragment contribution is

    N_F = sum_r w_F(r) rho(r) dV.

This is a smooth Voronoi-like partition, not a strict Voronoi/Bader analysis.
It can be applied to electron densities, density differences, or other scalar
fields stored in standard Gaussian Cube files.

Atom indices supplied on the command line are 1-based, matching common
chemistry conventions and Cube atom ordering.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CubeData:
    """Data read from a single Gaussian Cube file."""

    origin: np.ndarray
    shape: tuple[int, int, int]
    axes: np.ndarray
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Fragment:
    """Named fragment with zero-based atom indices."""

    name: str
    atom_indices: np.ndarray


def parse_cube(path: str | os.PathLike[str]) -> CubeData:
    """Read a standard single-dataset Gaussian Cube file.

    Notes
    -----
    Cube coordinates and grid vectors are used exactly as stored. Therefore,
    atom positions, grid coordinates, ``sigma``, and the volume element remain
    internally consistent regardless of whether the file uses bohr or angstrom.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        comment_1 = handle.readline()
        comment_2 = handle.readline()
        if not comment_1 or not comment_2:
            raise ValueError(f"{path}: incomplete Cube header")

        header = handle.readline().split()
        if len(header) < 4:
            raise ValueError(f"{path}: invalid atom-count/origin line")

        signed_n_atoms = int(header[0])
        n_atoms = abs(signed_n_atoms)
        origin = np.asarray(header[1:4], dtype=float)

        shape: list[int] = []
        axes: list[np.ndarray] = []
        for axis_number in range(3):
            parts = handle.readline().split()
            if len(parts) < 4:
                raise ValueError(f"{path}: invalid grid line {axis_number + 1}")
            n_points = abs(int(parts[0]))
            if n_points == 0:
                raise ValueError(f"{path}: grid dimension may not be zero")
            shape.append(n_points)
            axes.append(np.asarray(parts[1:4], dtype=float))

        atom_numbers = np.empty(n_atoms, dtype=int)
        atom_positions = np.empty((n_atoms, 3), dtype=float)
        for atom_index in range(n_atoms):
            parts = handle.readline().split()
            if len(parts) < 5:
                raise ValueError(f"{path}: invalid atom line {atom_index + 1}")
            atom_numbers[atom_index] = int(float(parts[0]))
            atom_positions[atom_index] = np.asarray(parts[2:5], dtype=float)

        # A negative atom count can indicate an additional dataset/orbital-ID
        # record. This implementation intentionally supports one scalar field.
        if signed_n_atoms < 0:
            dataset_line = handle.readline().split()
            if not dataset_line:
                raise ValueError(f"{path}: missing dataset identifier record")
            n_datasets = int(dataset_line[0])
            if n_datasets != 1:
                raise ValueError(
                    f"{path}: contains {n_datasets} datasets; only one is supported"
                )

        values = np.fromiter(
            (float(token) for line in handle for token in line.split()),
            dtype=float,
        )

    cube_shape = tuple(shape)
    expected = int(np.prod(cube_shape))
    if values.size != expected:
        raise ValueError(
            f"{path}: expected {expected} scalar values, found {values.size}"
        )

    return CubeData(
        origin=origin,
        shape=cube_shape,
        axes=np.vstack(axes),
        atom_numbers=atom_numbers,
        atom_positions=atom_positions,
        values=values,
    )


def parse_atom_selection(selection: str) -> np.ndarray:
    """Convert a 1-based selection such as ``1-6,9,12-14`` to zero-based indices."""
    indices: set[int] = set()
    for item in selection.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start < 1 or end < start:
                raise ValueError(f"invalid atom range: {item!r}")
            indices.update(range(start - 1, end))
        else:
            index = int(item)
            if index < 1:
                raise ValueError(f"atom indices must be positive: {item!r}")
            indices.add(index - 1)

    if not indices:
        raise ValueError("fragment atom selection is empty")
    return np.asarray(sorted(indices), dtype=int)


def parse_fragment(specification: str) -> Fragment:
    """Parse ``NAME:ATOM_SELECTION`` into a :class:`Fragment`."""
    if ":" not in specification:
        raise ValueError(
            f"invalid fragment {specification!r}; expected NAME:ATOM_SELECTION"
        )
    name, selection = specification.split(":", maxsplit=1)
    name = name.strip()
    if not name:
        raise ValueError("fragment name may not be empty")
    return Fragment(name=name, atom_indices=parse_atom_selection(selection))


def validate_fragments(fragments: Sequence[Fragment], n_atoms: int) -> None:
    """Validate names, atom ranges, and overlap among fragments."""
    names = [fragment.name for fragment in fragments]
    if len(set(names)) != len(names):
        raise ValueError("fragment names must be unique")

    assigned: dict[int, str] = {}
    for fragment in fragments:
        invalid = fragment.atom_indices[fragment.atom_indices >= n_atoms]
        if invalid.size:
            shown = ", ".join(str(index + 1) for index in invalid)
            raise ValueError(
                f"fragment {fragment.name!r} references atoms outside 1-{n_atoms}: {shown}"
            )
        for atom_index in fragment.atom_indices:
            atom_index = int(atom_index)
            if atom_index in assigned:
                raise ValueError(
                    f"atom {atom_index + 1} occurs in both {assigned[atom_index]!r} "
                    f"and {fragment.name!r}"
                )
            assigned[atom_index] = fragment.name


def sampled_grid(cube: CubeData, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled grid coordinates and flat indices into the Cube data array."""
    nx, ny, nz = cube.shape
    i = np.arange(0, nx, stride, dtype=np.int64)
    j = np.arange(0, ny, stride, dtype=np.int64)
    k = np.arange(0, nz, stride, dtype=np.int64)
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    ax, ay, az = cube.axes
    coordinates = (
        cube.origin
        + ii[:, None] * ax
        + jj[:, None] * ay
        + kk[:, None] * az
    )
    flat_indices = ii * (ny * nz) + jj * nz + kk
    return coordinates, flat_indices


def nearest_distances(
    coordinates: np.ndarray,
    atom_positions: np.ndarray,
    atom_indices: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Distance from every coordinate to the nearest selected atom."""
    selected_positions = atom_positions[atom_indices]
    result = np.empty(coordinates.shape[0], dtype=float)

    for start in range(0, coordinates.shape[0], chunk_size):
        stop = min(start + chunk_size, coordinates.shape[0])
        difference = (
            coordinates[start:stop, None, :] - selected_positions[None, :, :]
        )
        squared_distances = np.einsum("ijk,ijk->ij", difference, difference)
        result[start:stop] = np.sqrt(np.min(squared_distances, axis=1))
    return result


def smooth_fragment_weights(
    coordinates: np.ndarray,
    atom_positions: np.ndarray,
    fragments: Sequence[Fragment],
    sigma: float,
    chunk_size: int,
) -> np.ndarray:
    """Return normalized smooth distance weights with shape ``(n_fragments, n_grid)``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    distances = np.vstack(
        [
            nearest_distances(
                coordinates,
                atom_positions,
                fragment.atom_indices,
                chunk_size,
            )
            for fragment in fragments
        ]
    )

    # Subtract the largest exponent before exponentiation for numerical stability.
    exponents = -(distances**2) / (2.0 * sigma**2)
    exponents -= np.max(exponents, axis=0, keepdims=True)
    scores = np.exp(exponents)
    return scores / np.sum(scores, axis=0, keepdims=True)


def mean_grid_spacing(axes: np.ndarray) -> float:
    """Arithmetic mean of the three grid-vector lengths."""
    return float(np.mean(np.linalg.norm(axes, axis=1)))


def extract_step(path: str, pattern: str) -> int | None:
    """Extract a simulation step using the first capture group of a regex."""
    match = re.search(pattern, path)
    if match is None:
        return None
    if match.lastindex is None:
        raise ValueError("--step-regex must contain a capture group")
    return int(match.group(1))


def natural_sort_key(path: str) -> list[object]:
    """Sort strings naturally, so frame2 precedes frame10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate Cube scalar fields over arbitrary molecular fragments "
            "using smooth distance-based (Voronoi-like) weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Cube file(s) or quoted glob pattern(s), e.g. 'td.*/density.cube'",
    )
    parser.add_argument(
        "--fragment",
        action="append",
        required=True,
        metavar="NAME:ATOMS",
        help=(
            "Fragment definition with 1-based atom indices; repeat for each fragment. "
            "Examples: --fragment 'A1:1-6' --fragment 'D:7-20,33'"
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Absolute smoothing width in the Cube coordinate unit",
    )
    parser.add_argument(
        "--sigma-factor",
        type=float,
        default=3.0,
        help="Use sigma = factor x mean grid spacing when --sigma is omitted",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every Nth grid point along each axis (approximate for N > 1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of grid points processed per distance-calculation chunk",
    )
    parser.add_argument(
        "--step-regex",
        default=r"td\.(\d+)",
        help="Regex whose first capture group gives the simulation step",
    )
    parser.add_argument(
        "--reference",
        choices=("none", "first"),
        default="none",
        help="Also report fragment differences relative to the first sorted frame",
    )
    parser.add_argument(
        "--output",
        default="fragment_integrals.csv",
        help="Output CSV file",
    )
    return parser


def expand_inputs(patterns: Sequence[str]) -> list[str]:
    """Expand explicit paths and glob patterns, remove duplicates, and sort naturally."""
    matches: list[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            matches.extend(expanded)
        elif Path(pattern).is_file():
            matches.append(pattern)

    unique = sorted(set(matches), key=natural_sort_key)
    if not unique:
        raise FileNotFoundError("no Cube files matched the supplied input paths/patterns")
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.sigma is not None and args.sigma <= 0:
        parser.error("--sigma must be positive")
    if args.sigma_factor <= 0:
        parser.error("--sigma-factor must be positive")

    try:
        fragments = [parse_fragment(specification) for specification in args.fragment]
        files = expand_inputs(args.inputs)

        records: list[dict[str, object]] = []
        reference_values: dict[str, float] | None = None
        reference_structure: tuple[np.ndarray, np.ndarray] | None = None

        for frame_number, path in enumerate(files):
            cube = parse_cube(path)
            validate_fragments(fragments, len(cube.atom_numbers))

            if reference_structure is None:
                reference_structure = (cube.atom_numbers.copy(), cube.shape)
            else:
                reference_atom_numbers, reference_shape = reference_structure
                if not np.array_equal(cube.atom_numbers, reference_atom_numbers):
                    raise ValueError(f"{path}: atom ordering differs from the first Cube file")
                if cube.shape != reference_shape:
                    raise ValueError(f"{path}: grid shape differs from the first Cube file")

            coordinates, flat_indices = sampled_grid(cube, args.stride)
            sampled_values = cube.values[flat_indices]

            spacing = mean_grid_spacing(cube.axes)
            sigma = args.sigma if args.sigma is not None else args.sigma_factor * spacing
            weights = smooth_fragment_weights(
                coordinates,
                cube.atom_positions,
                fragments,
                sigma,
                args.chunk_size,
            )

            # For stride > 1 this is a rectangular-rule approximation.
            volume_element = abs(float(np.linalg.det(cube.axes))) * args.stride**3
            fragment_values = weights @ sampled_values * volume_element

            record: dict[str, object] = {
                "frame": frame_number,
                "step": extract_step(path, args.step_regex),
                "filename": Path(path).name,
                "grid_spacing": spacing,
                "sigma": sigma,
                "stride": args.stride,
            }
            for fragment, value in zip(fragments, fragment_values, strict=True):
                record[fragment.name] = float(value)
            record["total"] = float(np.sum(fragment_values))

            if reference_values is None:
                reference_values = {
                    fragment.name: float(value)
                    for fragment, value in zip(fragments, fragment_values, strict=True)
                }

            if args.reference == "first":
                for fragment, value in zip(fragments, fragment_values, strict=True):
                    record[f"delta_{fragment.name}"] = (
                        float(value) - reference_values[fragment.name]
                    )

            records.append(record)
            print(f"Processed {path}", file=sys.stderr)

        fragment_columns = [fragment.name for fragment in fragments]
        fieldnames = [
            "frame",
            "step",
            "filename",
            *fragment_columns,
            "total",
            "grid_spacing",
            "sigma",
            "stride",
        ]
        if args.reference == "first":
            fieldnames.extend(f"delta_{name}" for name in fragment_columns)

        with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    print(f"Wrote {args.output} ({len(records)} frame(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
