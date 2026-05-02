# Discrete Drainage Dynamics VIII — Cosmogenesis & DESI DR2

This directory contains the complete reproducibility package for the
manuscript

> **Discrete Drainage Dynamics VIII: Cosmogenesis from a Founding
> Impulse and the DESI DR2 Evolving Dark-Energy Signal**
> Stanislas de Wavrin, 2026.

The paper is part of the DDD series (Papers I–X). This volume shows
that a single localised exponentially-growing source on the discrete
drainage substrate of Paper~I produces a propagating cascade whose
effective equation of state crosses the phantom divide near
*z* ≈ 0.45, in agreement with the DESI DR2 evolving-*w* preference
at 2.8–4.2 σ.

## Repository contents

```
paperVIII_cosmogenesis/
├── README.md                    ← this file
├── LICENSE                      ← MIT license for the code
├── Makefile                     ← reproduce everything in one command
├── paper.tex                    ← LaTeX source (v9)
├── paper.pdf                    ← compiled manuscript (13 pages)
├── references.bib               ← BibTeX bibliography
├── cover_letter.txt             ← cover letter for journal submission
├── figures/                     ← publication-quality figures (PDF + PNG)
│   ├── fig_evolution            ← T²(t), I²(t), I²/T² ratio
│   ├── fig_m_eff                ← effective dilution exponent m_eff(t)
│   ├── fig_rho_DE               ← effective dark-energy density ρ_DE(z)
│   ├── fig_robustness           ← 10-realisation ensemble band
│   ├── fig_phase_heatmap        ← chronological T²(t,r) + m_eff(t) + w(z)
│   ├── fig_radial_profile       ← isotropic ⟨R(r)⟩ at 8 snapshots + (r,t) heatmap
│   ├── fig_iso_R_snapshots      ← 8 isotropic 2D maps of R(x,y,t)
│   └── fig_iso_T2_front         ← 8 isotropic 2D maps of T²(x,y,t) with matter front
├── code/                        ← reproducible simulation + analysis scripts
│   ├── 01_simulation.py         ← headline simulation (~30 s)
│   ├── 02_analysis.py           ← print diagnostics
│   ├── 03_make_figures.py       ← make fig_evolution, fig_m_eff, fig_rho_DE
│   ├── 04_robustness.py         ← 10-realisation ensemble (~5 min)
│   ├── 05_robustness_figure.py  ← make fig_robustness
│   ├── 06_finite_size_scaling.py← N = 4000 → 16000 scaling test
│   ├── 07_phase_heatmap.py      ← chronological 3-panel figure
│   ├── 09_radial_profile.py     ← isotropic radial-profile figure
│   └── 10_isotropic_snapshots.py← 8 snapshots: R(x,y) and T²(x,y) isotropic
└── data/                        ← cached numerical results
    ├── simulation_best.npz
    ├── ensemble.npz
    └── ensemble_summary.json
```

## Quick start

```bash
make all
```

This runs the headline simulation, generates all eight figures,
runs the 10-realisation ensemble, and compiles the manuscript.

```bash
make sim                # headline simulation only
make figures            # all single-realisation figures
make robustness         # full ensemble (~5 min)
make paper              # recompile manuscript only
make clean              # remove generated files
```

Direct Python invocation:

```bash
pip install numpy scipy matplotlib
cd code/
python 01_simulation.py
python 03_make_figures.py
python 07_phase_heatmap.py
python 09_radial_profile.py
python 10_isotropic_snapshots.py
```

## Headline result

| Quantity              | DESI DR2  | DDD simulation     | Agreement  |
|-----------------------|-----------|--------------------|------------|
| m_∞ (phantom phase)   | −3.72     | −3.72 ± 0.04       | 0.5 σ      |
| m_0 (today)           | +1.65     | +1.59 ± 0.18       | 0.3 σ      |
| w(z = 0)              | −0.45     | −0.47 ± 0.06       | 0.3 σ      |
| Phantom crossing z\*  | ~0.45     | ~0.45              | 0.4 %      |

Error bars are empirical standard deviations across 10 independent
realisations of the random Poisson-disk lattice
(`code/04_robustness.py`).

## What is being claimed (and what is not)

**Claim.** A single exponentially-growing source on a previously
empty DDD substrate produces a propagating cascade whose effective
*w*(z) reproduces the DESI DR2 evolving-dark-energy preference
quantitatively, with no DESI quantity fitted.

**Not claimed.** We do not derive the source pumping rate γ from
substrate first principles. We do not address the position of the
observer relative to the founding impulse Ω (deferred to Paper~X).
We do not claim that the dilution-vs-expansion ontological reading
of the result is itself proved by this paper — that programme is
the subject of Paper~IX.

## Connection with the rest of the DDD series

| Paper | Topic                                              | Status |
|-------|----------------------------------------------------|--------|
| I     | Substrate, drainage rule, emergent fields          | preprint |
| II    | Weak-field gravity, clock-rate suppression         | preprint |
| III   | Spinors, Weyl points, kinematic clock-rate         | preprint |
| IV    | Photon deflection, Shapiro delay                   | preprint |
| V     | Falsifiers (Yukawa, feedback, Eöt-Wash)            | preprint |
| VI    | Emergent gauge dynamics, fine-structure constant   | preprint |
| VII   | Gravity–gauge cross-coupling                       | preprint |
| **VIII** | **Cosmogenesis & DESI DR2 (this paper)**         | preprint |
| IX    | Effective cosmology, dilution-vs-expansion         | skeleton |
| X     | Off-centre observers, cosmological dipoles         | planned  |

## Reproducibility

All simulations use fixed random seeds (`np.random.seed(2024)` for
the headline run; `seed = 2024 + k * 1000` for realisation `k` in
the ensemble). Results should be reproducible to floating-point
precision on any standard scientific Python installation.

## License

* Code in `code/` is released under the MIT license (see `LICENSE`).
* Manuscript text and figures are © the author and made available
  under standard arXiv terms (free to read, redistribute, cite, with
  proper attribution).

## Contact

Stanislas de Wavrin
Independent researcher
sdewavrin@ohbibi.com
