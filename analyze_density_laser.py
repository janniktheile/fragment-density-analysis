#!/usr/bin/env python3
"""
Compare two difference-density CUBE series with molecular overlays and
Gaussian laser timelines.

Requirements
------------
numpy>=1.24
pandas>=2.0
matplotlib>=3.7

Typical usage
-------------
python analyze_density_laser_clean.py \
  --series-a \
    1moving_21.cube 1moving_35.cube 1moving_60.cube 1moving_105.cube \ #1 describes the cube-files of the 1 GW/cm^2 simulation
  --series-b \
    moving_21.cube moving_35.cube moving_60.cube moving_105.cube \
  --geo geo.xyz \
  --project-axis z \
  --profile-axis x \
  --labels "21 fs" "35 fs" "60 fs" "105 fs" \
  --laser-sigma 7 \
  --laser-t0 21 \
  --laser-start 10 \
  --laser-end 35 \
  --laser-height-a 0.045 \
  --laser-height-b 0.065 \
  --laser-amplitude-a 1 \
  --laser-amplitude-b 1.65 \
  --atom-size 180 \
  --zoom-padding 0.6 \
  --timeline-fig-width-per-panel 5.6 \
  --timeline-fig-height 10.0 \
  --timeline-offset 0.035 \
  --timeline-tick-size 30 \
  --timeline-label-size 24 \
  --legend-size 30 \
  --legend-title-size 32 \
  --laser-ymax 2.0 \
  --lower-timeline-legend-gap 0.055 \
  --shared-cbar-x 0.907 \
  --shared-cbar-y 0.135 \
  --colorbar-label-size 30 \
  --colorbar-tick-size 26 \
  --dpi 600 \
  --out-prefix moving_vsnmoving_higher_intens

Outputs
-------
<out-prefix>_2D_projected_two_series.png
<out-prefix>_fragment_charges_two_series.csv
<out-prefix>_analysis_summary_two_series.txt

Notes
-----
CUBE coordinates are assumed to be in Bohr and are converted to Angstrom.
Atom positions are taken from the CUBE headers; the XYZ file supplies element
symbols and atom order. Default fragments are A1=1-6, D=7-20, A2=21-32.
"""

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_VERSION = "v11-fixed-laser-scaling-and-legend-spacing"
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker

BOHR_TO_ANG = 0.529177210903

# Plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 38,
    "axes.labelsize": 38,
    "axes.titlesize": 38,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 30,
    "figure.dpi": 300,
})

FRAGMENTS_DEFAULT = {
    "A1": list(range(1, 7)),
    "D":  list(range(7, 21)),
    "A2": list(range(21, 33)),
}

FRAG_COLORS = {
    "A1": "#1b9e77",
    "D":  "#d95f02",
    "A2": "#7570b3",
}

ATOM_MARKERS = {
    "H": "o", "C": "o", "N": "^", "O": "D", "S": "s", "F": "P",
}

# Element colors are used for atom symbols. Fragment identity is shown by the
# colored fragment outlines/labels, so the atom colors do not encode fragments.
ELEMENT_COLORS = {
    "H": "white",
    "C": "#bdbdbd",
    "N": "#3b5bff",
    "O": "#e41a1c",
    "S": "#f1d54e",
    "F": "#4daf4a",
}

ELEMENT_LABEL_ORDER = ["H", "C", "N", "O", "S", "F"]


def style_axes(ax, rect_lw=3.0, spine_lw=3.0, tick_len=11, tick_w=2.8, tick_labelsize=26):
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_linewidth(spine_lw)
    ax.tick_params(axis="both", which="both", width=tick_w, length=tick_len,
                   direction="out", labelsize=tick_labelsize)


def add_panel_label(ax, letter):
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=36, color="black",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.65, linewidth=0.0),
            zorder=30)


@dataclass
class CubeData:
    path: str
    comments: tuple
    natoms: int
    origin: np.ndarray       # Angstrom
    axes: np.ndarray         # shape (3,3), Angstrom vectors
    shape: tuple             # (nx, ny, nz)
    atoms: list              # list of tuples (Z, charge, x,y,z) in Angstrom from cube
    data: np.ndarray         # shape (nx,ny,nz)


def unit_factor(unit):
    unit = unit.lower()
    if unit in ["bohr", "a.u.", "au"]:
        return BOHR_TO_ANG
    if unit in ["angstrom", "ang", "a"]:
        return 1.0
    raise ValueError("unit must be bohr or angstrom")


def read_cube(path, cube_units="bohr"):
    fac = unit_factor(cube_units)
    with open(path, "r") as f:
        c1 = f.readline().rstrip("\n")
        c2 = f.readline().rstrip("\n")
        parts = f.readline().split()
        natoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float) * fac
        shape = []
        axes = []
        for _ in range(3):
            p = f.readline().split()
            shape.append(abs(int(p[0])))
            axes.append([float(p[1]) * fac, float(p[2]) * fac, float(p[3]) * fac])
        axes = np.array(axes, dtype=float)
        atoms = []
        for _ in range(abs(natoms)):
            p = f.readline().split()
            Z = int(float(p[0]))
            chg = float(p[1])
            xyz = np.array([float(p[2]), float(p[3]), float(p[4])], dtype=float) * fac
            atoms.append((Z, chg, xyz[0], xyz[1], xyz[2]))
        # Remaining numbers are scalar values. np.fromstring is robust and fast enough here.
        raw = f.read()
    values = np.fromstring(raw, sep=" ", dtype=float)
    expected = shape[0] * shape[1] * shape[2]
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} cube values, found {values.size}")
    data = values.reshape(tuple(shape))
    return CubeData(path, (c1, c2), abs(natoms), origin, axes, tuple(shape), atoms, data)


def read_xyz(path, units="auto"):
    symbols = []
    coords = []
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    nat = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    if units == "auto":
        units_eff = "angstrom"
        if re.search(r"\bbohr\b|\bau\b|a\.u\.", comment, flags=re.I):
            units_eff = "bohr"
    else:
        units_eff = units
    fac = unit_factor(units_eff)
    for ln in lines[2:2+nat]:
        p = ln.split()
        symbols.append(p[0])
        coords.append([float(p[1]) * fac, float(p[2]) * fac, float(p[3]) * fac])
    return symbols, np.array(coords, dtype=float), units_eff


def grid_coordinates_1d(cube):
    """Return coordinates along cube axes for orthogonal grids."""
    # The uploaded cube is orthogonal. For non-orthogonal grids this still returns the coordinate
    # along the dominant component of each axis vector.
    coords = []
    for ax_i in range(3):
        vec = cube.axes[ax_i]
        dominant = int(np.argmax(np.abs(vec)))
        step_signed = vec[dominant]
        start = cube.origin[dominant]
        coords.append(start + np.arange(cube.shape[ax_i]) * step_signed)
    return coords


def axis_index(axis):
    return {"x": 0, "y": 1, "z": 2}[axis.lower()]


def get_spacing(cube):
    return np.array([np.linalg.norm(cube.axes[i]) for i in range(3)], dtype=float)


def project_2d(cube, project_axis="z"):
    ax = axis_index(project_axis)
    spacing = get_spacing(cube)
    proj = np.sum(cube.data, axis=ax) * spacing[ax]
    remaining = [i for i in range(3) if i != ax]
    coords = grid_coordinates_1d(cube)
    extent = [coords[remaining[0]][0], coords[remaining[0]][-1],
              coords[remaining[1]][0], coords[remaining[1]][-1]]
    # imshow expects array as [y, x]; transpose so first remaining coordinate is horizontal
    return proj.T, extent, remaining


def profile_1d(cube, profile_axis="x"):
    ax = axis_index(profile_axis)
    spacing = get_spacing(cube)
    other_axes = [i for i in range(3) if i != ax]
    prof = np.sum(cube.data, axis=tuple(other_axes)) * spacing[other_axes[0]] * spacing[other_axes[1]]
    coords = grid_coordinates_1d(cube)[ax]
    return coords, prof


def infer_label(path):
    base = os.path.basename(path).lower()
    if "start" in base:
        return "21 fs"
    m = re.search(r"(\d+(?:\.\d+)?)\s*fs", base, flags=re.I)
    if m:
        return f"{m.group(1)} fs"
    return os.path.splitext(os.path.basename(path))[0]


def molecule_limits(coords2d, padding=3.0):
    xmin, ymin = coords2d.min(axis=0)
    xmax, ymax = coords2d.max(axis=0)
    return xmin-padding, xmax+padding, ymin-padding, ymax+padding


def overlay_atoms_and_fragments(ax, symbols, xyz, remaining_axes, fragments, atom_size=70, label_offset=1.15):
    """Overlay atom symbols and fragment outlines on a projected 2D density map.

    Atom positions come from the cube header. The xyz file supplies only the
    element symbols. Fragment labels are placed above the corresponding fragment
    region to avoid covering the density signal.
    """
    xy = xyz[:, remaining_axes]

    # Soft fragment ellipses for orientation; labels above the fragments.
    for frag, atom_ids in fragments.items():
        idx = np.array(atom_ids, dtype=int) - 1
        pts = xy[idx]
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        w = max(1.0, float(pts[:, 0].max() - pts[:, 0].min() + 1.55))
        h = max(1.0, float(pts[:, 1].max() - pts[:, 1].min() + 1.35))

        ell = Ellipse(
            (cx, cy), width=w, height=h, angle=0.0,
            fill=False, linewidth=3.4, linestyle=(0, (4.0, 2.4)),
            edgecolor=FRAG_COLORS.get(frag, "black"),
            alpha=1.0, zorder=24
        )
        ax.add_patch(ell)

        # Put label above fragment, not in its center.
        label_y = float(pts[:, 1].max() + label_offset)
        label_txt = "A2(3F)" if frag == "A2" else frag
        label_artist = ax.text(
            cx, label_y, label_txt, fontsize=23, fontweight="bold",
            ha="center", va="bottom", color=FRAG_COLORS.get(frag, "black"),
            bbox=None,
            zorder=30,
        )
        label_artist.set_path_effects([
            pe.withStroke(linewidth=2.3, foreground="white")
        ])

    # Atoms: marker/colour by element. H is slightly smaller.
    for sym, (x, y) in zip(symbols, xy):
        sym = sym.capitalize()
        marker = ATOM_MARKERS.get(sym, "o")
        face = ELEMENT_COLORS.get(sym, "white")
        s = atom_size * (0.50 if sym == "H" else 1.0)
        ax.scatter(
            x, y, s=s, marker=marker, c=face, edgecolors="black",
            linewidths=0.9, alpha=0.98, zorder=18
        )


def add_delta_q_box(ax, row, fragments):
    """Horizontal per-panel box with projected fragment charges in 10^-3 e."""
    items = [
        TextArea(
            r"$\Delta q\;(10^{-3}\,e)$",
            textprops=dict(color="black", fontsize=21)
        )
    ]

    for frag in fragments:
        if frag in row and np.isfinite(row[frag]):
            label = "A2(3F)" if frag == "A2" else frag
            val = float(row[frag]) * 1e3
            col = FRAG_COLORS.get(frag, "black")

            items.append(
                TextArea(
                    rf"${label}={val:+.2f}$",
                    textprops=dict(color=col, fontsize=17)
                )
            )

    if len(items) == 1:
        return

    row_box = HPacker(
        children=items,
        align="center",
        pad=0,
        sep=13
    )

    anchored_box = AnchoredOffsetbox(
        loc="upper left",
        child=row_box,
        pad=0.30,
        frameon=True,
        bbox_to_anchor=(-0.2, 1.48),
        bbox_transform=ax.transAxes,
        borderpad=0.30
    )

    anchored_box.patch.set_boxstyle("round,pad=0.22")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_alpha(0.78)
    anchored_box.patch.set_linewidth(0.6)

    ax.add_artist(anchored_box)


def build_element_legend_handles(symbols):
    present = {s.capitalize() for s in symbols}
    order = [s for s in ELEMENT_LABEL_ORDER if s in present]
    order += sorted(present - set(order))
    handles = []
    for sym in order:
        handles.append(Line2D(
            [0], [0], marker=ATOM_MARKERS.get(sym, "o"), linestyle="None",
            color="black", markerfacecolor=ELEMENT_COLORS.get(sym, "white"),
            markeredgecolor="black", markeredgewidth=0.9, markersize=16, label=sym
        ))
    return handles


def build_fragment_legend_handles(fragments):
    return [Line2D(
        [0], [0], linestyle=(0, (4.0, 2.4)), linewidth=3.4,
        color=FRAG_COLORS.get(k, "black"),
        label=("A2(3F)" if k == "A2" else k)
    ) for k in fragments]

def global_projected_vmax(cubes, project_axis, percentile=None):
    vals = []
    for c in cubes:
        p, _, _ = project_2d(c, project_axis=project_axis)
        if percentile is None:
            vals.append(np.nanmax(np.abs(p)))
        else:
            vals.append(np.nanpercentile(np.abs(p), percentile))
    vmax = float(np.nanmax(vals))
    return vmax if vmax > 0 else 1.0


def compute_basic_integrals(cube, symbols, atom_xyz_cube, fragments, mol_padding=6.0, project_axis="z"):
    """Fast sanity checks plus fragment trend metrics.

    Fragment charges are computed from the 2D projected map using a nearest-atom
    Voronoi assignment in the projection plane. This is fast and consistent with
    the plotted 2D maps. Use it as a trend metric, not as a unique observable.
    """
    spacing = get_spacing(cube)
    voxel = float(np.prod(spacing))
    total = float(np.sum(cube.data) * voxel)
    pos = float(np.sum(cube.data[cube.data > 0]) * voxel)
    neg = float(np.sum(cube.data[cube.data < 0]) * voxel)
    absint = float(np.sum(np.abs(cube.data)) * voxel)

    # Leakage diagnostic using separable coordinate masks, without allocating all 3D points.
    coords1d = grid_coordinates_1d(cube)
    lo = atom_xyz_cube.min(axis=0) - mol_padding
    hi = atom_xyz_cube.max(axis=0) + mol_padding
    mx = (coords1d[0] >= lo[0]) & (coords1d[0] <= hi[0])
    my = (coords1d[1] >= lo[1]) & (coords1d[1] <= hi[1])
    mz = (coords1d[2] >= lo[2]) & (coords1d[2] <= hi[2])
    inside_mask = mx[:, None, None] & my[None, :, None] & mz[None, None, :]
    outside_abs = float(np.sum(np.abs(cube.data[~inside_mask])) * voxel)
    frac_outside = outside_abs / absint if absint != 0 else np.nan

    # 2D projected fragment charges. Projection already includes integration over project_axis.
    proj, extent, remaining = project_2d(cube, project_axis=project_axis)
    dx = abs((extent[1] - extent[0]) / (proj.shape[1] - 1)) if proj.shape[1] > 1 else 1.0
    dy = abs((extent[3] - extent[2]) / (proj.shape[0] - 1)) if proj.shape[0] > 1 else 1.0
    area = dx * dy
    xgrid = np.linspace(extent[0], extent[1], proj.shape[1])
    ygrid = np.linspace(extent[2], extent[3], proj.shape[0])
    X2, Y2 = np.meshgrid(xgrid, ygrid, indexing="xy")
    pts2 = np.column_stack([X2.ravel(), Y2.ravel()])
    atom2 = atom_xyz_cube[:, remaining]
    d2 = np.sum((pts2[:, None, :] - atom2[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)

    atom_to_frag = np.array([None] * len(symbols), dtype=object)
    for frag, ids in fragments.items():
        for atom_id in ids:
            atom_to_frag[atom_id - 1] = frag

    flat = proj.ravel()
    frag_charges = {}
    frag_abs = {}
    for frag in fragments:
        mask = atom_to_frag[nearest] == frag
        frag_charges[frag] = float(np.sum(flat[mask]) * area)
        frag_abs[frag + "_abs_projected"] = float(np.sum(np.abs(flat[mask])) * area)

    out = {
        "total_integral": total,
        "positive_integral": pos,
        "negative_integral": neg,
        "abs_integral": absint,
        "outside_abs_integral": outside_abs,
        "outside_abs_fraction": frac_outside,
    }
    out.update(frag_charges)
    out.update(frag_abs)
    return out


def make_2d_comparison(cubes, labels, symbols, atom_xyz_cube, fragments, args, charges_df=None):
    n = len(cubes)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0*ncols, 6.8*nrows), squeeze=False)

    vmax = args.vmax if args.vmax is not None else global_projected_vmax(
        cubes, args.project_axis, percentile=args.vmax_percentile)
    if args.relative_vmax is not None:
        vmax = vmax * args.relative_vmax

    im = None
    for i, (cube, label) in enumerate(zip(cubes, labels)):
        ax = axes.flat[i]
        proj, extent, remaining = project_2d(cube, project_axis=args.project_axis)
        im = ax.imshow(proj, origin="lower", extent=extent, cmap="seismic",
                       vmin=-vmax, vmax=vmax, interpolation=args.interpolation, aspect="equal")
        overlay_atoms_and_fragments(ax, symbols, atom_xyz_cube, remaining, fragments, label_offset=args.fragment_label_offset)
        if args.show_dq_box and charges_df is not None and i < len(charges_df):
            add_delta_q_box(ax, charges_df.iloc[i], fragments)
        if args.zoom:
            xy = atom_xyz_cube[:, remaining]
            xmin, xmax, ymin, ymax = molecule_limits(xy, padding=args.zoom_padding)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        ax.set_title(label, pad=14)
        ax.set_xlabel(f"{['x','y','z'][remaining[0]]} (Å)")
        ax.set_ylabel(f"{['x','y','z'][remaining[1]]} (Å)")
        add_panel_label(ax, f"({chr(97+i)})")
        style_axes(ax)

    for j in range(n, nrows*ncols):
        axes.flat[j].axis("off")

    # Shared colorbar. Tick labels are shown in units of 10^-3 for readability.
    cax = fig.add_axes([0.88, 0.24, 0.025, 0.52])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val*1e3:.1f}"))
    cbar.set_label(
    rf"Projected $\Delta\rho$ integrated along ${args.project_axis}$ "
    rf"($10^{{-3}}\ \mathrm{{e}}\,\AA^{{-2}}$)",
    labelpad=16
    )
    cbar.ax.tick_params(width=2.5, length=9, labelsize=24)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2.5)

    element_handles = build_element_legend_handles(symbols)
    fragment_handles = build_fragment_legend_handles(fragments)

    leg1 = fig.legend(
        handles=element_handles, title="Elements", ncol=len(element_handles),
        frameon=False, loc="lower center", bbox_to_anchor=(0.30, 0.035),
        fontsize=28, title_fontsize=30
    )
    leg1.get_frame().set_linewidth(2.0)
    leg1.get_frame().set_edgecolor("black")

    leg2 = fig.legend(
        handles=fragment_handles, title="Fragments", ncol=len(fragment_handles),
        frameon=False, loc="lower center", bbox_to_anchor=(0.80, 0.035),
        fontsize=28, title_fontsize=30
    )

    fig.suptitle(r"", y=0.985, fontsize=34)
    fig.subplots_adjust(left=0.10, right=0.82, top=0.90, bottom=0.17, wspace=0.45, hspace=args.panel_hspace)
    fig.savefig(args.out_prefix + "_2D_projected.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)



def _label_times_from_files(files, labels=None):
    if labels is not None:
        if len(labels) != len(files):
            raise ValueError("Number of labels must match number of cube files in the series.")
        return labels
    return [infer_label(path) for path in files]




def _label_times(labels):
    """Extract numerical snapshot times from labels such as '21 fs'."""
    times = []
    for label in labels:
        match = re.search(r"[-+]?\d*\.?\d+", str(label))
        if match:
            times.append(float(match.group()))
    return times


def _add_laser_timeline(fig, axes_row, args, labels, height, amplitude, y_max):
    """Place a Gaussian-pulse timeline directly below one panel row.

    Snapshot ticks are aligned with the horizontal centers of the corresponding
    molecular panels. Thus, the pulse maximum at t0 = 21 fs lies directly below
    the first snapshot rather than near the left edge of the full 10--105 fs axis.
    """
    valid_axes = [ax for ax in axes_row if ax.get_visible()]
    if not valid_axes:
        return None

    left = min(ax.get_position().x0 for ax in valid_axes)
    right = max(ax.get_position().x1 for ax in valid_axes)
    bottom = min(ax.get_position().y0 for ax in valid_axes)

    timeline_ax = fig.add_axes([
        left,
        bottom - args.timeline_offset - height,
        right - left,
        height,
    ])

    # Keep the timeline background fully transparent so its rectangular axes
    # patch cannot cover any part of the projected-density panels.
    timeline_ax.set_facecolor("none")
    timeline_ax.patch.set_alpha(0.0)
    timeline_ax.patch.set_visible(False)
    timeline_ax.set_zorder(0)

    snapshot_times = _label_times(labels)
    if len(snapshot_times) != len(valid_axes):
        raise ValueError("Each visible molecular panel needs one numerical snapshot time.")

    # Panel centers in the local coordinates of the timeline axis.
    panel_centers = np.array([
        ((ax.get_position().x0 + ax.get_position().x1) / 2.0 - left) / (right - left)
        for ax in valid_axes
    ])

    # Keep the 10 fs tick at the left edge, while aligning every snapshot time
    # exactly with its molecular panel. The right edge ends at the last snapshot.
    anchor_times = np.array([args.laser_start] + snapshot_times, dtype=float)
    anchor_x = np.concatenate(([0.0], panel_centers))

    def time_to_x(values):
        values = np.asarray(values, dtype=float)
        return np.interp(values, anchor_times, anchor_x)

    # Draw the pulse beyond 35 fs so its finite tail remains visible there.
    pulse_end = min(max(args.laser_end, snapshot_times[1] if len(snapshot_times) > 1 else args.laser_end),
                    snapshot_times[-1])
    pulse_t = np.linspace(args.laser_start, pulse_end, 1200)
    pulse_x = time_to_x(pulse_t)
    envelope = amplitude * np.exp(
        -0.5 * ((pulse_t - args.laser_t0) / args.laser_sigma) ** 2
    )

    timeline_ax.plot(pulse_x, envelope, color="0.25", linewidth=3.2)
    timeline_ax.fill_between(
        pulse_x, 0.0, envelope, color="0.75", alpha=0.45
    )

    timeline_ax.set_xlim(0.0, 1.0)
    # Use one fixed vertical scale for both timelines. Otherwise Matplotlib
    # rescales each pulse to its own maximum, making different amplitudes look
    # almost identical.
    timeline_ax.set_ylim(0.0, max(float(y_max), 1e-12))
    timeline_ax.set_yticks([])

    tick_times = [args.laser_start] + snapshot_times
    tick_x = time_to_x(tick_times)
    timeline_ax.set_xticks(tick_x)
    timeline_ax.set_xticklabels(
        [f"{t:g}" for t in tick_times],
        fontsize=args.timeline_tick_size,
    )

    timeline_ax.tick_params(
        axis="x",
        which="both",
        width=2.2,
        length=8,
        direction="out",
        pad=5,
    )

    for side in ["left", "right", "top", "bottom"]:
        timeline_ax.spines[side].set_visible(False)

    # Draw the time axis manually with an arrowhead at the right end.
    timeline_ax.annotate(
        "",
        xy=(1.018, 0.0),
        xytext=(0.0, 0.0),
        xycoords=timeline_ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=2.8,
            color="black",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=18,
        ),
        annotation_clip=False,
        zorder=40,
    )

    timeline_ax.text(
        1.035,
        0.0,
        r"$t$ [fs]",
        transform=timeline_ax.transAxes,
        ha="left",
        va="center",
        fontsize=args.timeline_label_size,
        clip_on=False,
    )

    return timeline_ax

def make_2d_two_series(series_a, series_b, labels, symbols, atom_xyz_cube, fragments, args):
    """Create a clean two-series figure with shared scale and compact timelines."""
    if len(series_a) != len(series_b):
        raise ValueError("--series-a and --series-b must contain the same number of cube files.")

    n = len(series_a)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(
            args.timeline_fig_width_per_panel * n,
            args.timeline_fig_height,
        ),
        squeeze=False,
        constrained_layout=False,
    )

    vmax_a = args.vmax_a if args.vmax_a is not None else global_projected_vmax(
        series_a, args.project_axis, percentile=args.vmax_percentile
    )
    vmax_b = args.vmax_b if args.vmax_b is not None else global_projected_vmax(
        series_b, args.project_axis, percentile=args.vmax_percentile
    )

    if args.relative_vmax is not None:
        vmax_a *= args.relative_vmax
        vmax_b *= args.relative_vmax

    # One shared colorbar requires one shared symmetric density scale.
    shared_vmax = max(vmax_a, vmax_b)

    row_specs = [
        (series_a, 0, "(a)"),
        (series_b, 1, "(b)"),
    ]
    shared_image = None

    for cubes, row, group_label in row_specs:
        for col, (cube, label) in enumerate(zip(cubes, labels)):
            ax = axes[row, col]
            ax.set_zorder(2)
            proj, extent, remaining = project_2d(
                cube, project_axis=args.project_axis
            )

            shared_image = ax.imshow(
                proj,
                origin="lower",
                extent=extent,
                cmap="seismic",
                vmin=-shared_vmax,
                vmax=shared_vmax,
                interpolation=args.interpolation,
                aspect="equal",
            )

            overlay_atoms_and_fragments(
                ax,
                symbols,
                atom_xyz_cube,
                remaining,
                fragments,
                atom_size=args.atom_size,
                label_offset=args.fragment_label_offset,
            )

            if args.zoom:
                xy = atom_xyz_cube[:, remaining]
                xmin, xmax, ymin, ymax = molecule_limits(
                    xy, padding=args.zoom_padding
                )
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            # Snapshot times are indicated by the shared timeline below each row.

            # Hide coordinate axes completely; only the molecular system remains.
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Reserve space below each row for its timeline and below the second
    # timeline for two non-overlapping legends.
    fig.subplots_adjust(
        left=args.fig_left,
        right=args.fig_right,
        top=args.fig_top,
        bottom=args.fig_bottom,
        wspace=args.panel_wspace,
        hspace=0.78,
    )

    for row, group_label in [(0, "(a)"), (1, "(b)")]:
        pos = axes[row, 0].get_position()
        fig.text(
            pos.x0 - args.group_label_dx,
            pos.y1 + args.group_label_dy,
            group_label,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=args.group_label_size,
            color="black",
        )

    # IMPORTANT: Keep one fixed shared y-scale for both pulse rows.
    # The previous automatic scale was proportional to the largest amplitude;
    # therefore increasing an amplitude also increased the axis limit and the
    # pulse appeared almost unchanged. A fixed --laser-ymax makes amplitude
    # changes directly visible and keeps both rows quantitatively comparable.
    shared_laser_ymax = float(args.laser_ymax)
    if shared_laser_ymax <= 0.0:
        raise ValueError("--laser-ymax must be greater than zero.")
    largest_amplitude = max(args.laser_amplitude_a, args.laser_amplitude_b)
    if largest_amplitude > shared_laser_ymax:
        raise ValueError(
            "The largest displayed laser amplitude exceeds --laser-ymax. "
            "Increase --laser-ymax to avoid clipping."
        )

    upper_timeline_ax = _add_laser_timeline(
        fig,
        axes[0, :],
        args,
        labels,
        height=args.laser_height_a,
        amplitude=args.laser_amplitude_a,
        y_max=shared_laser_ymax,
    )
    lower_timeline_ax = _add_laser_timeline(
        fig,
        axes[1, :],
        args,
        labels,
        height=args.laser_height_b,
        amplitude=args.laser_amplitude_b,
        y_max=shared_laser_ymax,
    )

    # One wider shared colorbar at the lower right.
    cax = fig.add_axes([
        args.shared_cbar_x,
        args.shared_cbar_y,
        args.shared_cbar_width,
        args.shared_cbar_height,
    ])
    cbar = fig.colorbar(shared_image, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f"{value * 1e3:.1f}")
    )
    cbar.set_label(
        rf"Projected $\Delta\rho$ integrated along ${args.project_axis}$"
        "\n"
        rf"($10^{{-3}}\ \mathrm{{e}}/\AA^2$)",
        labelpad=12,
        fontsize=args.colorbar_label_size,
    )
    cbar.ax.tick_params(
        width=2.2,
        length=8,
        labelsize=args.colorbar_tick_size,
    )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(2.2)

    element_handles = build_element_legend_handles(symbols)
    fragment_handles = build_fragment_legend_handles(fragments)

    # Place both legends a controlled distance below the lower time axis.
    # Using the top of each legend as the anchor makes the gap predictable.
    if lower_timeline_ax is None:
        raise RuntimeError("The lower laser timeline could not be created.")
    legend_anchor_y = (
        float(args.legend_y)
        if args.legend_y is not None
        else lower_timeline_ax.get_position().y0 - args.lower_timeline_legend_gap
    )

    fig.legend(
        handles=element_handles,
        title="Elements",
        ncol=len(element_handles),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(args.elements_legend_x, legend_anchor_y),
        fontsize=args.legend_size,
        title_fontsize=args.legend_title_size,
        columnspacing=0.65,
        handletextpad=0.25,
        borderaxespad=0.0,
        labelspacing=0.25,
    )

    fig.legend(
        handles=fragment_handles,
        title="Fragments",
        ncol=len(fragment_handles),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(args.fragments_legend_x, legend_anchor_y),
        fontsize=args.legend_size,
        title_fontsize=args.legend_title_size,
        columnspacing=0.85,
        handletextpad=0.35,
        borderaxespad=0.0,
        labelspacing=0.25,
    )

    fig.savefig(
        args.out_prefix + "_2D_projected_two_series.png",
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_delta_q_table(df, fragments):
    """Print compact Delta-q values in 10^-3 e for manual thesis tables."""
    display_cols = ["series", "label"] + list(fragments.keys())
    compact = df[display_cols].copy()
    for frag in fragments:
        compact[frag] = compact[frag] * 1e3
    print("\nProjected fragment charges for manual table (Delta q in 10^-3 e):")
    print(compact.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


def make_1d_profiles(cubes, labels, args, atom_xyz_cube=None):
    fig, ax = plt.subplots(figsize=(14.5, 8.5))

    for cube, label in zip(cubes, labels):
        x, y = profile_1d(cube, profile_axis=args.profile_axis)
        ax.plot(x, y, linewidth=3.0, label=label)

    if atom_xyz_cube is not None:
        ai = axis_index(args.profile_axis)
        xmin = float(atom_xyz_cube[:, ai].min() - args.zoom_padding)
        xmax = float(atom_xyz_cube[:, ai].max() + args.zoom_padding)
        ax.set_xlim(xmin, xmax)

    ax.axhline(0.0, color="black", linewidth=1.2, alpha=1.0)
    ax.set_xlabel(f"{args.profile_axis} coordinate (Å)")
    other = ''.join([a for a in 'xyz' if a != args.profile_axis])
    ax.set_ylabel(rf"$\iint \Delta\rho(x,y,z)\,d{other[0]}\,d{other[1]}$ (e/Å)")
    ax.set_title(f"1D charge redistribution profile along {args.profile_axis}", pad=18)
    ax.legend(frameon=False, ncol=2, loc="best")
    style_axes(ax)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.18)
    fig.savefig(args.out_prefix + "_1D_profiles.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_fragments(text):
    # Format: A1:1-6,D:7-20,A2:21-32
    if not text:
        return FRAGMENTS_DEFAULT
    frags = {}
    for item in text.split(','):
        name, rng = item.split(':')
        ids = []
        for part in rng.split('+'):
            if '-' in part:
                a, b = part.split('-')
                ids.extend(range(int(a), int(b)+1))
            else:
                ids.append(int(part))
        frags[name.strip()] = ids
    return frags


def main():
    print(f"Running analyze_density_laser_clean.py [{SCRIPT_VERSION}]")
    p = argparse.ArgumentParser(description="Projected difference-density analysis with geometry overlay.")
    p.add_argument("cubes", nargs="*", help="Fallback: one-series cube files, kept for backward compatibility")
    p.add_argument("--series-a", nargs="+", default=None, help="First cube series, e.g. moving nuclei")
    p.add_argument("--series-b", nargs="+", default=None, help="Second cube series, e.g. fixed nuclei")
    p.add_argument("--geo", required=True, help="Geometry xyz file used for the calculation")
    p.add_argument("--project-axis", default="z", choices=["x", "y", "z"], help="Out-of-plane direction to integrate over")
    p.add_argument("--profile-axis", default="x", choices=["x", "y", "z"], help="Axis for 1D profile")
    p.add_argument("--cube-units", default="bohr", choices=["bohr", "angstrom"], help="Coordinate units in cube file")
    p.add_argument("--geo-units", default="auto", choices=["auto", "bohr", "angstrom"], help="Coordinate units in xyz file")
    p.add_argument("--fragments", default="A1:1-6,D:7-20,A2:21-32", help="Fragment definition, e.g. A1:1-6,D:7-20,A2:21-32")
    p.add_argument("--labels", nargs="*", default=None, help="Optional labels/time labels; for two-series mode they apply to both series")
    p.add_argument("--out-prefix", default="moving_vs_nonmoving", help="Prefix for output files")
    p.add_argument("--zoom", action="store_true", default=True, help="Zoom to molecule region")
    p.add_argument("--no-zoom", dest="zoom", action="store_false", help="Do not zoom to molecule")
    p.add_argument("--zoom-padding", type=float, default=0.6, help="Padding around molecule for zoom, in Angstrom")
    p.add_argument("--vmax", type=float, default=None, help="Manual symmetric colorbar max for projected maps in one-series mode")
    p.add_argument("--vmax-a", type=float, default=None, help="Manual symmetric colorbar max for series A only")
    p.add_argument("--vmax-b", type=float, default=None, help="Manual symmetric colorbar max for series B only")
    p.add_argument("--vmax-percentile", type=float, default=99.5, help="Percentile for robust global colorbar max; use 100 for true max")
    p.add_argument("--relative-vmax", type=float, default=None, help="Optional factor applied to global vmax, e.g. 0.8")
    p.add_argument("--interpolation", default="bilinear", choices=["nearest", "bilinear", "bicubic"], help="imshow interpolation")
    p.add_argument("--mol-padding", type=float, default=6.0, help="Padding for leakage bounding box in Angstrom")
    p.add_argument("--panel-hspace", type=float, default=0.37, help="Vertical spacing between subplot rows")
    p.add_argument("--fragment-label-offset", type=float, default=1.25, help="Vertical offset for fragment labels above atoms, in Angstrom")
    p.add_argument("--show-dq-box", action="store_true", default=False, help="Show projected fragment charges inside each 2D panel in one-series mode")
    p.add_argument("--no-dq-box", dest="show_dq_box", action="store_false", help="Hide projected fragment-charge boxes")
    p.add_argument("--atom-size", type=float, default=180, help="Atom marker size in 2D plots")
    p.add_argument("--time-label-size", type=float, default=30, help="Font size for time labels above subpanels")
    p.add_argument("--group-label-size", type=float, default=42, help="Font size for (a)/(b) labels")
    p.add_argument("--group-label-dx", type=float, default=0.060, help="Horizontal offset for (a)/(b) labels in figure coordinates")
    p.add_argument("--group-label-dy", type=float, default=0.006, help="Vertical offset for (a)/(b) labels in figure coordinates")
    p.add_argument("--axis-label-size", type=float, default=30, help="Font size for axis labels")
    p.add_argument("--tick-label-size", type=float, default=26, help="Font size for tick labels")
    p.add_argument("--legend-size", type=float, default=30, help="Font size for legends")
    p.add_argument("--legend-title-size", type=float, default=32, help="Font size for legend titles")
    p.add_argument("--colorbar-label-size", type=float, default=30, help="Font size for colorbar label")
    p.add_argument("--colorbar-tick-size", type=float, default=26, help="Font size for colorbar ticks")
    p.add_argument("--panel-wspace", type=float, default=0.07, help="Horizontal spacing between subplot columns")
    p.add_argument("--fig-left", type=float, default=0.075, help="Figure left margin")
    p.add_argument("--fig-right", type=float, default=0.82, help="Figure right margin; leave space for vertical colorbars")
    p.add_argument("--fig-top", type=float, default=0.985, help="Figure top margin")
    p.add_argument("--fig-bottom", type=float, default=0.18, help="Figure bottom margin")
    p.add_argument("--cbar-x", type=float, default=0.86, help="x-position of both vertical colorbars")
    p.add_argument("--cbar-width", type=float, default=0.022, help="width of vertical colorbars")
    p.add_argument("--cbar-height", type=float, default=0.3, help="height of each vertical colorbar")
    p.add_argument("--cbar-a-y", type=float, default=0.65, help="y-position of colorbar for series A")
    p.add_argument("--cbar-b-y", type=float, default=0.175, help="y-position of colorbar for series B")
    p.add_argument("--legend-y", type=float, default=None, help="Optional manual y-position for the top of both legends; by default it is derived from the lower timeline")
    p.add_argument("--lower-timeline-legend-gap", type=float, default=0.055, help="Vertical gap in figure coordinates between the lower time axis and the top of the legends")
    p.add_argument("--separator-x0", type=float, default=0.035, help="left x-position of separator line between series, figure coordinates")
    p.add_argument("--separator-x1", type=float, default=0.905, help="right x-position of separator line between series, figure coordinates")
    p.add_argument("--separator-lw", type=float, default=4, help="line width of separator between series")
    p.add_argument("--separator-alpha", type=float, default=0.95, help="opacity of separator line between series")
    p.add_argument("--laser-sigma", type=float, default=7.0, help="Gaussian pulse width sigma in fs")
    p.add_argument("--laser-t0", type=float, default=21.0, help="Center of the Gaussian pulse in fs")
    p.add_argument("--laser-start", type=float, default=10.0, help="Start of the laser timeline in fs")
    p.add_argument("--laser-end", type=float, default=35.0, help="End of the laser timeline in fs")
    p.add_argument("--laser-height-a", type=float, default=0.045, help="Figure-coordinate height of the series-A laser timeline")
    p.add_argument("--laser-height-b", type=float, default=0.07, help="Figure-coordinate height of the series-B laser timeline")
    p.add_argument("--laser-amplitude-a", type=float, default=1.0, help="Displayed laser-envelope amplitude for series A")
    p.add_argument("--laser-amplitude-b", type=float, default=1.65, help="Displayed laser-envelope amplitude for series B")
    p.add_argument("--laser-ymax", type=float, default=2.0, help="Fixed shared vertical maximum for both laser timelines; keep this constant when comparing displayed amplitudes")
    p.add_argument("--timeline-fig-width-per-panel", type=float, default=5.6, help="Figure width per time panel in inches")
    p.add_argument("--timeline-fig-height", type=float, default=10.0, help="Total figure height in inches")
    p.add_argument("--timeline-offset", type=float, default=0.035, help="Vertical gap between each panel row and its timeline")
    p.add_argument("--timeline-tick-size", type=float, default=30, help="Font size of timeline tick labels")
    p.add_argument("--timeline-label-size", type=float, default=24, help="Font size of the timeline axis label")
    p.add_argument("--shared-cbar-x", type=float, default=0.907, help="x-position of the shared colorbar")
    p.add_argument("--shared-cbar-y", type=float, default=0.135, help="y-position of the shared colorbar")
    p.add_argument("--shared-cbar-width", type=float, default=0.016, help="width of the shared colorbar")
    p.add_argument("--shared-cbar-height", type=float, default=0.44, help="height of the shared colorbar")
    p.add_argument("--elements-legend-x", type=float, default=0.31, help="horizontal center of the Elements legend")
    p.add_argument("--fragments-legend-x", type=float, default=0.68, help="horizontal center of the Fragments legend")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    fragments = parse_fragments(args.fragments)
    xyz_symbols, xyz_coords, xyz_units = read_xyz(args.geo, units=args.geo_units)

    two_series_mode = args.series_a is not None or args.series_b is not None
    if two_series_mode:
        if args.series_a is None or args.series_b is None:
            raise ValueError("For two-series mode, provide both --series-a and --series-b.")
        if args.cubes:
            raise ValueError("Use either positional cubes OR --series-a/--series-b, not both.")
        if len(args.series_a) != len(args.series_b):
            raise ValueError("--series-a and --series-b must contain the same number of cube files.")

        series_a = [read_cube(path, cube_units=args.cube_units) for path in args.series_a]
        series_b = [read_cube(path, cube_units=args.cube_units) for path in args.series_b]
        labels = _label_times_from_files(args.series_a, args.labels)
        all_cubes = series_a + series_b

        atom_xyz_cube = np.array([[a[2], a[3], a[4]] for a in all_cubes[0].atoms], dtype=float)
        if len(xyz_symbols) != len(atom_xyz_cube):
            raise ValueError(f"Number of atoms differs: xyz has {len(xyz_symbols)}, cube has {len(atom_xyz_cube)}")
        symbols = xyz_symbols

        max_cube_atom_shift = 0.0
        for cube in all_cubes[1:]:
            xyz_this = np.array([[a[2], a[3], a[4]] for a in cube.atoms], dtype=float)
            if xyz_this.shape != atom_xyz_cube.shape:
                raise ValueError(f"Cube {cube.path} has a different atom count/shape.")
            max_cube_atom_shift = max(max_cube_atom_shift, float(np.max(np.linalg.norm(xyz_this - atom_xyz_cube, axis=1))))

        rmsd_no_align = float(np.sqrt(np.mean((atom_xyz_cube - xyz_coords) ** 2))) if atom_xyz_cube.shape == xyz_coords.shape else np.nan

        rows = []
        for series_name, cubes, paths in [("a", series_a, args.series_a), ("b", series_b, args.series_b)]:
            for cube, label, path in zip(cubes, labels, paths):
                vals = compute_basic_integrals(cube, symbols, atom_xyz_cube, fragments,
                                               mol_padding=args.mol_padding, project_axis=args.project_axis)
                vals["series"] = series_name
                vals["label"] = label
                vals["cube_file"] = os.path.basename(path)
                rows.append(vals)

        df = pd.DataFrame(rows)
        cols = ["series", "label", "cube_file"] + [c for c in df.columns if c not in ["series", "label", "cube_file"]]
        df = df[cols]
        df.to_csv(args.out_prefix + "_fragment_charges_two_series.csv", index=False)

        make_2d_two_series(series_a, series_b, labels, symbols, atom_xyz_cube, fragments, args)
        print_delta_q_table(df, fragments)

        with open(args.out_prefix + "_analysis_summary_two_series.txt", "w") as f:
            f.write("Difference-density two-series analysis\n")
            f.write("====================================\n\n")
            f.write(f"Series a files: {len(series_a)}\n")
            f.write(f"Series b files: {len(series_b)}\n")
            f.write(f"Geometry: {args.geo} (read as {xyz_units}; used for symbols/order only)\n")
            f.write(f"Cube units assumed: {args.cube_units}; CUBE atom coordinates used for overlay/fragment partition\n")
            f.write(f"Projection: integrated along {args.project_axis}\n")
            f.write(f"Fragments: {fragments}\n")
            f.write(f"RMSD between cube atom positions and xyz positions without alignment (diagnostic only): {rmsd_no_align:.6f} Å\n")
            f.write(f"Maximum atom-position shift between cube files: {max_cube_atom_shift:.6e} Å\n")
            f.write("\nSanity checks / integrals:\n")
            f.write(df.to_string(index=False))
            f.write("\n\nNote: A1/D/A2 fragment charges use a nearest-atom Voronoi partition of the 2D projected map.\n")

        print("\nWrote:")
        print(f"  {args.out_prefix}_2D_projected_two_series.png")
        print(f"  {args.out_prefix}_fragment_charges_two_series.csv")
        print(f"  {args.out_prefix}_analysis_summary_two_series.txt")
        return

    # Backward-compatible one-series mode.
    if not args.cubes:
        raise ValueError("Provide cube files positionally, or use --series-a and --series-b.")

    cubes = [read_cube(path, cube_units=args.cube_units) for path in args.cubes]
    labels = args.labels if args.labels else [infer_label(path) for path in args.cubes]
    if len(labels) != len(cubes):
        raise ValueError("Number of --labels must match number of cube files")

    atom_xyz_cube = np.array([[a[2], a[3], a[4]] for a in cubes[0].atoms], dtype=float)

    if len(xyz_symbols) != len(atom_xyz_cube):
        raise ValueError(
            f"Number of atoms differs: xyz has {len(xyz_symbols)}, cube has {len(atom_xyz_cube)}"
        )

    symbols = xyz_symbols

    max_cube_atom_shift = 0.0
    for cube in cubes[1:]:
        xyz_this = np.array([[a[2], a[3], a[4]] for a in cube.atoms], dtype=float)
        if xyz_this.shape != atom_xyz_cube.shape:
            raise ValueError(f"Cube {cube.path} has a different atom count/shape.")
        max_cube_atom_shift = max(max_cube_atom_shift, float(np.max(np.linalg.norm(xyz_this - atom_xyz_cube, axis=1))))

    if atom_xyz_cube.shape == xyz_coords.shape:
        rmsd_no_align = float(np.sqrt(np.mean((atom_xyz_cube - xyz_coords) ** 2)))
    else:
        rmsd_no_align = np.nan

    rows = []
    for cube, label in zip(cubes, labels):
        vals = compute_basic_integrals(cube, symbols, atom_xyz_cube, fragments, mol_padding=args.mol_padding, project_axis=args.project_axis)
        vals["label"] = label
        vals["cube_file"] = os.path.basename(cube.path)
        rows.append(vals)
    df = pd.DataFrame(rows)
    cols = ["label", "cube_file"] + [c for c in df.columns if c not in ["label", "cube_file"]]
    df = df[cols]
    df.to_csv(args.out_prefix + "_fragment_charges.csv", index=False)

    make_2d_comparison(cubes, labels, symbols, atom_xyz_cube, fragments, args, charges_df=df)
    make_1d_profiles(cubes, labels, args, atom_xyz_cube=atom_xyz_cube)

    with open(args.out_prefix + "_analysis_summary_series.txt", "w") as f:
        f.write("Difference-density series analysis\n")
        f.write("================================\n\n")
        f.write(f"Cube files: {len(cubes)}\n")
        f.write(f"Geometry: {args.geo} (read as {xyz_units}; used for symbols/order only)\n")
        f.write(f"Cube units assumed: {args.cube_units}; CUBE atom coordinates used for overlay/fragment partition; output coordinates plotted in Angstrom\n")
        f.write(f"Projection: integrated along {args.project_axis}\n")
        f.write(f"1D profile axis: {args.profile_axis}\n")
        f.write(f"Fragments: {fragments}\n")
        f.write(f"RMSD between cube atom positions and xyz positions without alignment (diagnostic only): {rmsd_no_align:.6f} Å\n")
        f.write(f"Maximum atom-position shift between cube files: {max_cube_atom_shift:.6e} Å\n")
        f.write("\nSanity checks / integrals:\n")
        f.write(df.to_string(index=False))
        f.write("\n\nInterpretation hints:\n")
        f.write("- total_integral should be close to zero for |LUMO|^2-|HOMO|^2 if both orbitals are normalized.\n")
        f.write("- positive_integral and negative_integral should have similar absolute values.\n")
        f.write("- outside_abs_fraction is a simple diagnostic for diffuse/leaking density outside an atom bounding box.\n")
        f.write("- A1/D/A2 fragment charges use a nearest-atom Voronoi partition; use them as a consistent trend metric, not as an absolute observable.\n")

    print_delta_q_table(df.assign(series=""), fragments)
    print("Wrote:")
    print(f"  {args.out_prefix}_2D_projected.png")
    print(f"  {args.out_prefix}_1D_profiles.png")
    print(f"  {args.out_prefix}_fragment_charges.csv")
    print(f"  {args.out_prefix}_analysis_summary_series.txt")


if __name__ == "__main__":
    main()
