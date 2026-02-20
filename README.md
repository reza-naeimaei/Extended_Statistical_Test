# Interval-extended congruency tests (TLS deformation monitoring)

This repository contains the simulation code and the corresponding paper for **classical** vs **interval-extended** congruency tests that explicitly account for *remaining systematic effects* as **unknown-but-bounded** biases.

**Paper (IVK 2026 proceedings):**
- *Beyond a Pure Stochastic Treatment: Integrating Remaining Systematics into Congruency Tests*  
  Reza Naeimaei, Steffen Schön  
  DOI: `10.3217/978-3-99161-070-0-013` (see [`Paper/ivk_2026_013.pdf`](Paper/ivk_2026_013.pdf))

## Repository structure

- `1D Case/` – 1D simulations (classical and interval-extended)
- `2D Case/` – 2D simulations (classical and interval-extended BOX/Zonotope)
- `Paper/` – final PDF corresponding to the simulation study

Each simulation folder contains an `outputs/` directory with figures and (for 2D) cached maps (`.npz`) used in the paper.

## Requirements

- Python **>= 3.9**
- Packages: `numpy`, `scipy`, `matplotlib`, `tqdm`, `joblib`

Install with pip:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the simulations

### 1D
```bash
python 1D/01_classic_1D.py
python 1D/02_extended_box_1D.py
```

### 2D
```bash
python 2D/01_classic_2D.py
python 2D/02_extended_box_2D.py
```

The scripts write figures to the corresponding `*/outputs/...` folders.

## Notes on reproducibility

- The scripts use fixed RNG seeds (see the `Config...` dataclasses) for reproducible Monte Carlo results.
- The 2D scripts can be computationally heavier; they support parallel execution via `joblib`.

## License

- **Code:** MIT License (see `LICENSE`).
- **Paper PDF:** includes its own license statement inside the document (CC BY 4.0 for the paper content, with exclusions for third-party material as noted in the PDF).
