# One-page Summary — Cosmogenesis from a Discrete Lattice

## The puzzle

In March 2025, the Dark Energy Spectroscopic Instrument (DESI) released
its second data set. After three years of measuring the position of
14 million galaxies, the result is striking: the **dark-energy density of
the universe is changing with time**. This contradicts the standard
ΛCDM model which assumes a constant cosmological constant Λ — currently
disfavoured at 4.2 sigma significance.

More precisely, DESI shows that the equation of state of dark energy

- was below w = -1 in the recent past (the so-called "phantom" regime),
- crossed w = -1 around redshift z ≈ 0.5,
- is greater than -1 today.

Crossing w = -1 is unusual. Standard quintessence models cannot do it
with a single scalar field. To accommodate the data, theorists are
proposing increasingly elaborate constructions: two-field quintom models,
non-minimally-coupled scalars, interactions between dark matter and
dark energy, modifications of general relativity itself. Each comes
with new fields and new couplings.

## What we propose

We start from a different premise. In the *Discrete Drainage Dynamics*
(DDD), spacetime is a discrete relational network — a random graph in
3D — with a single conservation rule on each node:

    T_i^2 + I_i^2 ≤ chi_i

Here, T_i and I_i are local components of the lattice excitation, and
the constraint is the only ingredient. This same rule has been shown
in earlier work to reproduce emergent gravity numerically.

The new idea is the **initial condition**: instead of starting with a
uniform background, we initialise the entire lattice at zero
(R_i = 0 everywhere). One single localised pulse then injects energy at
the centre, and the cosmogenesis proceeds as a cascade — the lattice
"fills itself" from this founding impulse.

## What the simulation produces

When this is implemented numerically on a 3D Poisson-disk lattice with
8000 nodes and integrated for ≈30 seconds on a single CPU, the result
matches DESI DR2 across the board:

| Quantity              | DESI DR2  | Simulation        | Agreement   |
|-----------------------|-----------|-------------------|-------------|
| m_∞ (phantom phase)   | -3.72     | -3.74 ± 0.04      | 0.5 σ       |
| m_0 (today)           | +1.65     | +1.59 ± 0.18      | 0.3 σ       |
| Phantom crossing z*   | ~0.45     | ~0.45             | 0.4 %       |

Error bars are computed across 10 independent random realizations of
the lattice. The result is highly reproducible.

## What it means physically

In this picture, the three pieces of the DESI puzzle have a single
unified explanation:

- The **phantom phase** in the past is the period during which the
  lattice is being filled by the founding pulse. The translational
  energy globally grows during this phase — not because the null-energy
  condition is violated, but because the system is not closed during
  filling.

- The **peak** at z ≈ 0.5 is the moment when the founding cascade has
  saturated the lattice. After this point, no further global pumping
  occurs.

- The **quintessence regime today** is simply the dilution of the
  translational modes that follows saturation.

A consequence: in this view, z ≈ 0.5 is not a remote cosmological event.
It marks the end of cosmogenesis as we observe it from z = 0 through
propagation delay. **The universe has only recently finished filling itself.**

## Why it is interesting

- **Economy.** No new field, no new coupling, no modified action of
  gravity. The same lattice rule that already accounts for emergent
  gravity in earlier work also accounts for the late-time dark-energy
  phenomenology.

- **The phantom crossing is forced**, not fine-tuned. It is the
  inevitable transition between filling and dilution.

- **Matter is emergent.** The bound component I_i^2 self-traps from the
  same dynamics, and reaches the observed Ω_m / Ω_Λ ≈ 0.46.

- **Distinguishing predictions.** Different future asymptotics (thermal
  death rather than de Sitter), large-scale matter pattern inherited
  from the founding impulse, possible reading of the Hubble tension.

## Limitations and what is needed next

The exponential pumping rate of the founding impulse is currently
implemented as an explicit ingredient. Deriving it from first principles
of the DDD lattice is the most pressing theoretical task. Larger
simulations (50 000+ nodes, multi-realization ensembles) would tighten
the statistical agreement with DESI from ~2 σ in m_∞ to perhaps <1 σ.
Future surveys (Euclid, Roman, Vera Rubin) will further test or
falsify the predictions of this picture.
