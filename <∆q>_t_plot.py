#!/usr/bin/env python3
"""
Plot fragment-resolved charge redistribution and its time average.

Requirements
------------
numpy>=2
pandas>=2.0
matplotlib>=3.7
scipy>=1.10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# User settings
csv_path = "put_in_your_path.csv"

step_col = "step"
col_A1 = "NA1"
col_D = "ND"
col_A2 = "NA2"

col_dA1 = "dNA1"
col_dD = "dND"
col_dA2 = "dNA2"

interpret_as_steps = True
dt_fs = 0.001

interp_dt_fs = 0.05

draw_pulse = True
pulse_center_fs = 21.0
pulse_sigma_fs = 7.0

colors = {
    "A1": "#1b9e77",
    "D": "#d95f02",
    "A2": "#7570b3",
    "A1+A2": "#e7298a",
}

line_width = 3.0


def compute_time_average(time, values):
    """Return the time-weighted average of a curve."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(time) & np.isfinite(values)
    time = time[valid]
    values = values[valid]

    if len(time) < 2 or time[-1] <= time[0]:
        raise ValueError("At least two increasing time points are required.")

    return np.trapezoid(values, time) / (time[-1] - time[0])


def interpolate_curve(time, values, dt):
    """Return a shape-preserving interpolated curve."""
    time_new = np.arange(time[0], time[-1] + dt, dt)
    values_new = PchipInterpolator(time, values)(time_new)
    return time_new, values_new


def draw_gaussian_pulse(ax, time_min, time_max, y_limits):
    """Draw the Gaussian laser-pulse envelope above zero."""
    time = np.linspace(time_min, time_max, 2000)
    pulse = np.exp(
        -0.5 * ((time - pulse_center_fs) / pulse_sigma_fs) ** 2
    )

    height = 0.16 * (y_limits[1] - y_limits[0])
    pulse *= height

    ax.fill_between(
        time,
        0.0,
        pulse,
        color="black",
        alpha=0.12,
        zorder=0,
    )
    ax.plot(
        time,
        pulse,
        color="black",
        linewidth=1.4,
        zorder=1,
    )


def style_axes(ax):
    """Apply common figure formatting."""
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    ax.tick_params(
        axis="both",
        direction="out",
        width=2.2,
        length=9,
        labelsize=18,
    )


df = pd.read_csv(csv_path)

columns_to_convert = [
    step_col,
    col_A1,
    col_D,
    col_A2,
    col_dA1,
    col_dD,
    col_dA2,
]

for column in columns_to_convert:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

df = (
    df.dropna(subset=[step_col])
    .sort_values(step_col)
    .reset_index(drop=True)
)

if interpret_as_steps:
    time_fs = df[step_col].to_numpy(dtype=float) * dt_fs
else:
    time_fs = df[step_col].to_numpy(dtype=float)


delta_columns = [col_dA1, col_dD, col_dA2]

if all(column in df.columns for column in delta_columns):
    df = df.dropna(subset=delta_columns).reset_index(drop=True)

    if interpret_as_steps:
        time_fs = df[step_col].to_numpy(dtype=float) * dt_fs
    else:
        time_fs = df[step_col].to_numpy(dtype=float)

    delta_A1 = df[col_dA1].to_numpy(dtype=float)
    delta_D = df[col_dD].to_numpy(dtype=float)
    delta_A2 = df[col_dA2].to_numpy(dtype=float)

else:
    absolute_columns = [col_A1, col_D, col_A2]
    missing = [
        column for column in absolute_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=absolute_columns).reset_index(drop=True)

    if interpret_as_steps:
        time_fs = df[step_col].to_numpy(dtype=float) * dt_fs
    else:
        time_fs = df[step_col].to_numpy(dtype=float)

    q_A1 = df[col_A1].to_numpy(dtype=float)
    q_D = df[col_D].to_numpy(dtype=float)
    q_A2 = df[col_A2].to_numpy(dtype=float)

    delta_A1 = q_A1 - q_A1[0]
    delta_D = q_D - q_D[0]
    delta_A2 = q_A2 - q_A2[0]


curves = (
    delta_A1,
    delta_D,
    delta_A2,
    delta_A1 + delta_A2,
)

titles = (
    "A1",
    "D",
    "A2(3F)",
    "A1 + A2(3F)",
)

panel_labels = ("(a)", "(b)", "(c)", "(d)")

curve_colors = (
    colors["A1"],
    colors["D"],
    colors["A2"],
    colors["A1+A2"],
)

interpolated_curves = [
    interpolate_curve(time_fs, curve, interp_dt_fs)
    for curve in curves
]

time_averages = [
    compute_time_average(time_fs, curve)
    for curve in curves
]


all_plot_values = [
    values for _, values in interpolated_curves
]

y_min = min(np.nanmin(values) for values in all_plot_values)
y_max = max(np.nanmax(values) for values in all_plot_values)

padding = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
y_limits = (y_min - padding, y_max + padding)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 140,
})


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()


for ax, curve, average, title, panel, color in zip(
    axes,
    interpolated_curves,
    time_averages,
    titles,
    panel_labels,
    curve_colors,
):
    plot_time, plot_values = curve

    ax.set_xlim(time_fs[0], time_fs[-1])
    ax.set_ylim(y_limits)

    if draw_pulse:
        draw_gaussian_pulse(
            ax,
            time_fs[0],
            time_fs[-1],
            y_limits,
        )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        zorder=1,
    )

    ax.plot(
        plot_time,
        plot_values,
        color=color,
        linewidth=line_width,
        zorder=3,
    )

    ax.axhline(
        average,
        color=color,
        linestyle="--",
        linewidth=2.4,
        zorder=4,
    )

    ax.text(
        0.97,
        0.92,
        rf"$\langle\Delta q\rangle_t"
        rf" = {average:+.3f}\,\mathrm{{e}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=19,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.75,
            "linewidth": 0.0,
        },
        zorder=5,
    )

    ax.text(
        0.03,
        0.96,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
    )

    ax.set_title(title, pad=14)
    style_axes(ax)


axes[0].set_ylabel(r"$\Delta q$ (e)")
axes[2].set_ylabel(r"$\Delta q$ (e)")

axes[2].set_xlabel("Time (fs)")
axes[3].set_xlabel("Time (fs)")


fig.subplots_adjust(
    left=0.11,
    right=0.97,
    top=0.91,
    bottom=0.11,
    wspace=0.22,
    hspace=0.30,
)

plt.show()

# fig.savefig(
#     "time_averaged_charge_redistribution.png",
#     dpi=600,
#     bbox_inches="tight",
# )