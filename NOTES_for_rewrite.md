# Notes for Paper VIII rewrite

Last updated: 2026-05-02

## Paragraphs / framings to integrate when rewriting

### 1. Geometric interpretation of cosmic acceleration (§Physical picture)

To insert in §Physical picture (probably as first or second paragraph after "What the substrate does"):

> *Why does this look like accelerating expansion?* As the cascade
> front sweeps outward through the discrete lattice, the geometric
> area available to it grows: at radius $r_{\rm front}(t)$ the front
> contacts $\propto r_{\rm front}^2(t)$ nodes simultaneously. Even
> with a strictly bounded local transfer rate, the global drainage
> throughput therefore scales geometrically with the front's
> surface --- and an internal observer at $\Omega$, who calibrates
> time against the $\langle T^2\rangle$ peak, reconstructs this
> geometric growth as an effective acceleration of expansion.
> **No fundamental repulsive force is required**: the apparent
> dark-energy phantom phase is the operational signature of a
> finite-resource cascade reaching ever-more substrate; the
> effective deceleration after saturation is the signature of that
> supply running out.

### 2. Branching as candidate origin of γ (§Open questions)

Paragraph to add in §Open questions, replacing or expanding "Derivation of γ":

> *Branching origin of γ.* A natural candidate mechanism for
> the exponential growth rate $\gamma$ is a substrate-level
> branching process: each newly activated node activates its
> neighbours at a rate $\beta$ depending on local connectivity
> and on a substrate activation threshold, giving a population
> $N_{\rm active}(t) = N_0 e^{\beta t}$ in the early-cascade
> regime. If $\beta$ can be derived from the local structure of
> the lattice (mean degree, activation threshold, drainage
> coupling), then $\gamma$ becomes a *prediction* of the
> framework rather than a parameter. Numerical tests of this
> mechanism on the DDD substrate are reported in
> Appendix~X / [companion code].

### 3. Source-profile robustness (new §Sensitivity subsection)

After running the source profile scan with N=8000 and corrected T formula:

- All monotone-growing-and-shutting-off profiles produce the same
  qualitative phantom-to-quintessence pattern
- m_0 ≈ +1.6 universally — drainage-rule signature, profile-independent
- m_∞ depends on profile shape: only exponential and (with mild
  tension) linear match DESI's m_∞ = -3.72 closely; constant,
  quadratic, cubic, sigmoid are excluded
- The exponential remains the empirical best fit but is no longer
  privileged "by construction" — it survives the test

### 4. Bug-fix note (internal)

Two figure-generating scripts (07_phase_heatmap.py and
10_isotropic_snapshots.py) used a simplified approximation
T = max(net_flux, 0) instead of the paper's stated definition
T_i = (alpha/D_avg) * sum_{j~i} max(R_j - R_i, 0). This gives
qualitatively correct but quantitatively off m_eff(t) trajectories.
Fixed in v11.5 of the codebase. Headline simulation
(04_robustness.py) was always correct and reproduces DESI exactly.

## Decisions made

- Keep the exponential profile as headline (Option A from the
  source profile scan discussion). Linear is at 2.6σ tension on
  m_inf and is mentioned as alternative parameter-free candidate.
- Add §Source-profile robustness as new subsection or appendix.
- Defer the dilution-vs-expansion derivation to Paper IX.

## Open paths

- If the branching test (12_branching_test.py? to be named) gives
  N_active ~ e^(βt) with β ≈ 0.415, this would derive γ from
  substrate dynamics --- principal achievement for v12.
- If not, document the negative result and keep γ as the principal
  open question.
