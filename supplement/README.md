# Reported configurations

This folder holds the sphere centres of every configuration the paper reports, together with a script that checks them. The script needs only Python 3 and numpy, not the solver.

```
python verify_packing.py configurations                                  # all of them
python verify_packing.py configurations/stoyan-yaskov-8/euclidean/p80.csv # one
```

For each file, the script recomputes from the centres and the container's faces the largest radius r at which the configuration is feasible:

    r = min( min_{i<j} d(c_i, c_j) / 2 ,  min_i d(c_i, boundary) ),

where d is the file's metric (l2, l1 or l-inf), and every centre must lie inside the container. It then prints r beside the claimed radius. Balls of radius r centred at the c_i then have disjoint interiors and lie inside the container.

## Contents

| path | what |
|---|---|
| `configurations/<container>/<metric>/p<p>.csv` | Best seed of every cell in the main experiment (240 cells) and in the Stoyan & Yaskov comparison (129 cells: l2, l1, l-inf) |
| `configurations/variants/escape/...` | Runs continued with the opt-in escape phase: Stoyan & Yaskov example 6 at p = 30–34, the H-box under l-inf at p = 56, and the cage under l2 at p = 100 (7 cells). These differ from the default solver; see the paper's escape-phase paragraph |
| `configurations/h-box/sup_norm/p56.csv` | The H-box under l-inf at p = 56, a cell outside the main grid (results/hbox_sup_56.csv) |
| `configurations.csv` | Index: study, container, metric, p, radius, seed, file |
| `containers/*.json` | Each container as a list of planar faces, given as vertex rings with optional holes, each with a unit normal pointing into the container |
| `verify_packing.py` | The checker |

Each configuration file starts with `#` lines giving:

- the container and its face list (`geometry`, relative to the file);
- the metric, `p` and the claimed `radius`;
- the run it came from (results file, seed and solver commit).

Then comes `i,x,y,z`, one row per centre. Values are written at full double precision: each one reads back as exactly the stored number.

## How the check works

- **Inside-ness.** A ray is cast from each centre and its crossings with the faces are counted; an odd count means the centre is inside. This is done along five generic directions, and all five must agree.
- **Distance to a face.** This is exact in all three norms. The nearest point of a face lies in one of two places:
  - inside the face, where the norm's ball first touches the face's plane. That distance is |n·c − b| / ‖n‖\*, using the dual norm.
  - on one of the face's edges. Point-to-segment distances are computed in closed form: by projection for l2, and from the breakpoints of the piecewise-linear objective for l1 and l-inf.
- **Tolerance.** A configuration passes when r ≥ claimed × (1 − 1e-9). The solver shrinks the radius it reports by a relative 1e-9 so that it is safely feasible, so the checker's r comes out about 1e-9 above the claimed value.
- **Result.** All 377 configurations pass (240 + 129 + 1 for the H-box under l-inf at p = 56 + 7 from the escape phase: example 6 at p = 30–34, that H-box cell, and the cage under l2 at p = 100), each with r between 0.99998e-9 and 1.0000008e-9 (relative) above its claimed radius.

## Regenerating

```
python -m tsgo.experiments.export_configurations   # from the repository root
```

This reads `results/sweep.csv`, `results/stoyan_yaskov.csv`, `results/stoyan_yaskov_escape.csv`, `results/hbox_sup_56.csv`, `results/hbox_sup_56_escape.csv` and `results/cage_100_escape.csv`, plus the run files under `output/`. It refuses to export a run whose file does not hold the radius its table row names.
