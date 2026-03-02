# Refinement Channel

Observer-dependent convergence of Dirac manifolds in twisted Kagome bilayers.

## Scripts

| Script | Method | Description |
|--------|--------|-------------|
| `bpdm_kagome.py` | Direct selection | Top-m eigenstates by Dirac score |
| `bpdm_markov.py` | Viterbi path | Globally smoothest projector path via DP |
| `bpdm_evolved.py` | Evolved basis | Parallel-transport + swap tracking |
| `bpdm_minimal_m.py` | Minimal subspace | Iterative manifold growth |
| `bpdm_minimal_m2.py` | Minimal v2 | Refined minimal construction |
| `bpdm_ultra.py` | Ultra-converged | Extended shells up to N=7 |
| `bpdm_fast.py` | Fast kernel | Kernel projection method |
| `bpdm_coarse.py` | **Refinement channel** | Partial-trace coarse-graining (key result) |

## Key Result

For the m=6 Dirac manifold at θ=9.6°:
- Raw overlap: S_raw = 3.6 × 10⁻⁵
- Coarse-grained: S_coarse = 2.0 (55,000× improvement)

## Requirements

- Python 3.8+
- NumPy, SciPy

## Author

D. J. Edwards, Swansea University
