# Fragment Density Analysis

Python tools for fragment-resolved charge analysis, density-difference calculations, and visualization of real-time TDDFT simulations using Gaussian Cube files.

## Overview

This repository contains Python scripts developed for the analysis of time-dependent electron-density redistribution in molecular donor–acceptor systems.

The scripts cover two complementary analysis workflows:

1. generation and visualization of projected density-difference maps at selected propagation times;
2. fragment-resolved integration of time-dependent electron densities using a smooth distance-based real-space partitioning scheme.

The tools were developed for real-time time-dependent density functional theory (RT-TDDFT) simulations of the asymmetric A1–D–A2(3F) oligomer, but several scripts can also be applied to other molecular systems after adapting the input files and fragment definitions.

## Repository Structure

```text
fragment-density-analysis/
│
├── density_difference.py
├── analyze_density_laser.py
├── fragment_cube_partition.py
├── delta_q_t_plot.py
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Requirements

The scripts require Python 3 and the following packages:

```text
numpy>=2.0
pandas>=2.0
matplotlib>=3.7
scipy>=1.10
```

The dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

# Workflow 1: Projected Density-Difference Maps

This workflow is used to generate two-dimensional projected density-difference maps of the A1–D–A2(3F) oligomer at selected propagation times during and after the laser pulse.

The Gaussian laser-pulse profiles are displayed together with the molecular snapshots. Different pulse intensities can be represented by Gaussian profiles of different heights.

The density-difference maps are integrated along the selected projection direction, which is typically the molecular out-of-plane (z) direction.

## Step 1: Generate the Density-Difference Cube Files

The script

```text
density_difference.py
```

calculates a normalized density difference between two wavefunction Cube files:

[
\Delta \rho(\mathbf r)
======================

## \left|\psi_{\mathrm{LUMO}}^{\mathrm{norm}}(\mathbf r)\right|^2

\left|\psi_{\mathrm{HOMO}}^{\mathrm{norm}}(\mathbf r)\right|^2.
]

Both input wavefunctions are normalized before the difference is calculated:

[
\int
\left|\psi^{\mathrm{norm}}(\mathbf r)\right|^2
,d\mathbf r
===========

1.

]

The input and output file paths are defined near the beginning of the script:

```python
HOMO = Path("wf-st0069.cube")
LUMO = Path("wf-st0070.cube")
OUT = Path("difference_density_normalized.cube")
SUMMARY = Path("difference_density_normalized_summary.txt")
```

The script must be applied separately to the corresponding Cube files for each selected propagation time.

For example, density-difference Cube files may be generated for snapshots at:

```text
21 fs
35 fs
60 fs
105 fs
```

The output consists of:

```text
difference_density_normalized.cube
difference_density_normalized_summary.txt
```

The summary file reports the original wavefunction norms, the normalized density integrals, the extrema of the density difference, and its total integral.

## Step 2: Plot the Projected Density-Difference Maps

The resulting density-difference Cube files are used as input for:

```text
analyze_density_laser.py
```

This script generates projected density-difference maps for one or two simulation series.

A typical comparison may include:

* a lower-intensity simulation;
* a higher-intensity simulation;
* moving-nuclei and frozen-nuclei simulations;
* selected propagation times during and after the laser pulse.

The volumetric density difference is integrated along the selected projection axis:

[
\Delta \rho_{\mathrm{proj}}(x,y)
================================

\int \Delta \rho(x,y,z),dz.
]

For the A1–D–A2(3F) oligomer, the projection is typically performed along the (z) direction.

A typical command is:

```bash
python analyze_density_laser.py \
  --series-a \
    intensity1_21.cube \
    intensity1_35.cube \
    intensity1_60.cube \
    intensity1_105.cube \
  --series-b \
    intensity2_21.cube \
    intensity2_35.cube \
    intensity2_60.cube \
    intensity2_105.cube \
  --geo geometry.xyz \
  --project-axis z \
  --labels "21 fs" "35 fs" "60 fs" "105 fs" \
  --laser-t0 21 \
  --laser-sigma 7 \
  --laser-amplitude-a 1.0 \
  --laser-amplitude-b 1.65 \
  --laser-ymax 2.0 \
  --dpi 600 \
  --out-prefix density_comparison
```

The principal figure output is:

```text
density_comparison_2D_projected_two_series.png
```

The script can additionally generate analysis summaries and fragment-resolved trend metrics.

The projected fragment values calculated within this visualization script are based on a two-dimensional nearest-atom assignment and should primarily be interpreted as qualitative trend metrics. For the quantitative fragment-resolved population analysis, use `fragment_cube_partition.py`.

---

# Workflow 2: Fragment-Resolved Charge Redistribution

This workflow is used to quantify the time-dependent charge redistribution among user-defined molecular fragments.

For the A1–D–A2(3F) oligomer, the molecular structure is divided into:

```text
A1
D
A2(3F)
```

## Smooth Real-Space Fragment Partitioning

To quantify dynamic and directional charge redistribution without discontinuities associated with sharp grid boundaries, the script employs a smooth real-space partitioning scheme using distance-based soft weights.

Each grid point (\mathbf r) is assigned continuously to a molecular fragment (k) through the Gaussian weighting function

[
w_k(\mathbf r)
==============

\frac{
\exp\left[-d_k^2(\mathbf r)/(2\sigma^2)\right]
}{
\sum_j
\exp\left[-d_j^2(\mathbf r)/(2\sigma^2)\right]
},
]

where

[
d_k(\mathbf r)
==============

\min_{\alpha\in k}
\left|\mathbf r-\mathbf R_\alpha\right|
]

is the Euclidean distance between the grid point and the nearest atom (\alpha) belonging to fragment (k).

The parameter (\sigma) controls the spatial smoothness of the fragment boundaries.

By construction, the weights satisfy the partition-of-unity condition

[
\sum_k w_k(\mathbf r)=1
]

at every point in space.

The time-dependent fragment populations are calculated through a discrete summation over the real-space grid:

[
q_k(t)
\approx
\sum_i
w_k(\mathbf r_i)
\rho(\mathbf r_i,t)
\Delta V,
]

where (\Delta V) is the grid-voxel volume.

This approach provides a numerically stable representation of continuous charge flow across molecular subunits.

The method is a smooth distance-based or Voronoi-like partitioning scheme. It is not a strict Voronoi, Bader, or topological atoms-in-molecules analysis.

## Calculating Fragment Populations

The script

```text
fragment_cube_partition.py
```

integrates the electron density from a series of Gaussian Cube files over arbitrary user-defined fragments.

Fragment definitions are supplied through the command line using 1-based atom indices.

For the A1–D–A2(3F) oligomer, a typical command is:

```bash
python fragment_cube_partition.py \
  "td.*/density.cube" \
  --fragment "A1:1-6" \
  --fragment "D:7-20" \
  --fragment "A2:21-32" \
  --reference first \
  --output fragment_integrals.csv
```

The fragment assignments in this example are:

```text
A1: atoms 1–6
D: atoms 7–20
A2(3F): atoms 21–32
```

The script supports arbitrary fragment names and atom selections. For example:

```bash
--fragment "Donor:1-18"
--fragment "Bridge:19-26"
--fragment "Acceptor:27-43"
```

The smoothing width can either be supplied directly:

```bash
--sigma 0.9
```

or determined relative to the average grid spacing:

```bash
--sigma-factor 3.0
```

If `--reference first` is specified, the script additionally calculates the population change relative to the first processed frame:

[
\Delta q_k(t)
=============

q_k(t)-q_k(t_0).
]

The output CSV file contains columns such as:

```text
frame
step
filename
A1
D
A2
total
grid_spacing
sigma
stride
delta_A1
delta_D
delta_A2
```

The total column provides a useful check of electron-number conservation across the simulation.

---

# Workflow 3: Plotting Time-Dependent Fragment Charges

The output CSV file generated by `fragment_cube_partition.py` can be visualized using:

```text
delta_q_t_plot.py
```

The script plots the time-dependent fragment-resolved charge redistribution:

[
\Delta q_{\mathrm{A1}}(t),
\qquad
\Delta q_{\mathrm D}(t),
\qquad
\Delta q_{\mathrm{A2}}(t).
]

It can additionally display the combined acceptor contribution:

[
\Delta q_{\mathrm{A1}}(t)
+
\Delta q_{\mathrm{A2}}(t).
]

The script also evaluates the time-weighted average:

[
\left\langle \Delta q_k \right\rangle_t
=======================================

\frac{
\int_{t_\mathrm{start}}^{t_\mathrm{end}}
\Delta q_k(t),dt
}{
t_\mathrm{end}-t_\mathrm{start}
}.
]

Before running the script, the input CSV path and column names can be adapted in the user-settings section:

```python
csv_path = "fragment_integrals.csv"

step_col = "step"

col_dA1 = "delta_A1"
col_dD = "delta_D"
col_dA2 = "delta_A2"
```

Depending on the CSV format, the column names may instead be:

```python
col_dA1 = "dNA1"
col_dD = "dND"
col_dA2 = "dNA2"
```

The conversion from propagation steps to time is controlled through:

```python
interpret_as_steps = True
dt_fs = 0.001
```

The figure can be saved by enabling the `savefig` command at the end of the script:

```python
fig.savefig(
    "time_averaged_charge_redistribution.png",
    dpi=600,
    bbox_inches="tight",
)
```

---

# Recommended Analysis Order

For projected density-difference figures:

```text
wavefunction Cube files
        ↓
density_difference.py
        ↓
density-difference Cube files
        ↓
analyze_density_laser.py
        ↓
2D projected density-difference figure
```

For fragment-resolved charge dynamics:

```text
time-dependent density Cube files
        ↓
fragment_cube_partition.py
        ↓
fragment population CSV
        ↓
delta_q_t_plot.py
        ↓
time-dependent fragment charge plots
```

---

# Notes

* Gaussian Cube files must use a consistent atom ordering.
* Fragment atom indices follow the atom order of the Cube files.
* Atom indices supplied to `fragment_cube_partition.py` are 1-based.
* All Cube files within one analysis should use compatible grids and molecular structures.
* The correct propagation time step must be specified when converting simulation steps into femtoseconds.
* The density-difference and fragment-population workflows represent complementary analyses and should not be interpreted as identical quantities.
* Large Cube files should normally not be committed directly to GitHub. Example inputs or reduced test data may be included instead.
* File paths and system-specific parameters should be adapted before running the scripts.

## Citation

If you use these scripts in academic work, please cite the associated publication:

> **Strong-Field-Driven Non-Linear Vibronic Coupling in an Asymmetric Donor-Acceptor Oligomer**
> Citation and DOI to be added after publication.

## License

This project is distributed under the MIT License.
