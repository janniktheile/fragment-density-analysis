#!/usr/bin/env python3

from pathlib import Path
import numpy as np #here density difference from dominant HOMO (69) --> LUMO (70) transition


HOMO = Path("wf-st0069.cube")
LUMO = Path("wf-st0070.cube")
OUT = Path("difference_density_normalized.cube")
SUMMARY = Path("difference_density_normalized_summary.txt")


def read_cube(path: Path):
    with path.open("r") as f:
        lines = f.readlines()

    comment1 = lines[0].rstrip("\n")
    comment2 = lines[1].rstrip("\n")
    hdr = lines[2].split()
    natoms = int(hdr[0])
    origin = tuple(map(float, hdr[1:4]))

    grid = []
    for i in range(3, 6):
        parts = lines[i].split()
        grid.append((int(parts[0]), tuple(map(float, parts[1:4]))))

    atom_lines = [line.rstrip("\n") for line in lines[6:6 + abs(natoms)]]
    data = np.fromstring(" ".join(lines[6 + abs(natoms):]), sep=" ")

    nx, ny, nz = [g[0] for g in grid]
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(f"{path.name}: expected {expected} values, found {data.size}")

    return {
        "comment1": comment1,
        "comment2": comment2,
        "natoms": natoms,
        "origin": origin,
        "grid": grid,
        "atom_lines": atom_lines,
        "data": data.reshape((nx, ny, nz)),
    }


def write_cube(path: Path, template: dict, data: np.ndarray, comment1: str, comment2: str):
    with path.open("w") as f:
        f.write(comment1 + "\n")
        f.write(comment2 + "\n")
        f.write(f"{template['natoms']:5d} {template['origin'][0]:12.6f} {template['origin'][1]:12.6f} {template['origin'][2]:12.6f}\n")
        for n, vec in template["grid"]:
            f.write(f"{n:5d} {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
        for line in template["atom_lines"]:
            f.write(line + "\n")

        flat = data.ravel(order="C")
        for i in range(0, flat.size, 6):
            chunk = flat[i:i+6]
            f.write(" ".join(f"{x:13.5e}" for x in chunk) + "\n")


if not HOMO.exists():
    raise FileNotFoundError(f"HOMO-Datei nicht gefunden: {HOMO}")
if not LUMO.exists():
    raise FileNotFoundError(f"LUMO-Datei nicht gefunden: {LUMO}")

cube_homo = read_cube(HOMO)
cube_lumo = read_cube(LUMO)

if cube_homo["natoms"] != cube_lumo["natoms"]:
    raise ValueError("Unterschiedliche Atomanzahl.")
if cube_homo["origin"] != cube_lumo["origin"] or cube_homo["grid"] != cube_lumo["grid"]:
    raise ValueError("Unterschiedliches Grid oder unterschiedliche Origin.")

grid_vectors = np.array([g[1] for g in cube_homo["grid"]], dtype=float)
dV = abs(np.linalg.det(grid_vectors))

psi_homo = cube_homo["data"]
psi_lumo = cube_lumo["data"]

rho_homo = psi_homo**2
rho_lumo = psi_lumo**2

norm_homo = float(rho_homo.sum() * dV)
norm_lumo = float(rho_lumo.sum() * dV)

if norm_homo <= 0 or norm_lumo <= 0:
    raise ValueError("Nichtpositive Norm gefunden.")

psi_homo_n = psi_homo / np.sqrt(norm_homo)
psi_lumo_n = psi_lumo / np.sqrt(norm_lumo)

rho_homo_n = psi_homo_n**2
rho_lumo_n = psi_lumo_n**2
drho_n = rho_lumo_n - rho_homo_n

write_cube(
    OUT,
    cube_lumo,
    drho_n,
    comment1="Normalized difference density from wavefunction cubes",
    comment2="Computed as |psi_LUMO_norm|^2 - |psi_HOMO_norm|^2 with both wavefunctions normalized to 1",
)

stats = {
    "voxel_volume_bohr3": float(dV),
    "norm_homo_before": norm_homo,
    "norm_lumo_before": norm_lumo,
    "integral_rho_homo_n": float(rho_homo_n.sum() * dV),
    "integral_rho_lumo_n": float(rho_lumo_n.sum() * dV),
    "drho_min": float(drho_n.min()),
    "drho_max": float(drho_n.max()),
    "drho_mean": float(drho_n.mean()),
    "drho_rms": float(np.sqrt(np.mean(drho_n**2))),
    "integral_drho": float(drho_n.sum() * dV),
}

with SUMMARY.open("w") as f:
    f.write("Normalized difference density summary\n")
    f.write("Computed as |psi_LUMO_norm|^2 - |psi_HOMO_norm|^2\n")
    f.write("Both wavefunctions were normalized so that integral |psi|^2 dV = 1.\n\n")
    for k, v in stats.items():
        f.write(f"{k}: {v:.10e}\n")

print("Erstellt:")
print(OUT)
print(SUMMARY)
