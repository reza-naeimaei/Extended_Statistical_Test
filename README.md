# Interval-extended congruency tests

This repository contains the simulation code and the corresponding paper for **classical** vs **interval-extended** congruency tests that explicitly account for *remaining systematic effects* as **unknown-but-bounded** biases. 

Classical congruency tests assess significance based on a purely stochastic model, which can lead to overly optimistic results when systematic effects are significant.

Here, we demonstrate via simulations how neglecting remaining systematic errors affects classical decisions, and how incorporating them through an interval-extended test (with **box** or **zonotope** bias models) leads to robust and interpretable decisions in both **1D** and **2D** cases.

**Paper (IVK 2026 proceedings):**
- *Beyond a Pure Stochastic Treatment: Integrating Remaining Systematics into Congruency Tests*  
  Reza Naeimaei, Steffen Schön  
  DOI: `10.3217/978-3-99161-070-0-013` (see [`Paper/ivk_2026_013.pdf`](Paper/ivk_2026_013.pdf))

## Repository structure

- `1D Case/` – 1D simulations (classical and interval-extended)
- `2D Case/` – 2D simulations (classical and interval-extended box/zonotope)
- `Paper/` – final PDF corresponding to the simulation study

Each simulation folder contains an `outputs/` directory with figures and cached result files (`.npz`) produced by the scripts.

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

## Geometric intuition: Minkowski sum and difference (2D)

In the 2D interval-extended congruency test, remaining systematic effects are modeled as an admissible bounded set \(B\) (axis-aligned **box** or **zonotope**). The classical acceptance region is the ellipse
\(E = \{\mathbf d : \mathbf d^\top \Sigma_d^{-1}\mathbf d \le k_\alpha\}\).
The extended decision regions follow directly from Minkowski operations: the robust outer boundary is \(A_{\mathrm{ext}} = E \oplus B\), the strict-accept region is \(A_{\mathrm{in}} = E \ominus B\), and the ambiguity region is \(A_{\mathrm{amb}} = A_{\mathrm{ext}} \setminus A_{\mathrm{in}}\).
See `docs/minkowski/` for animations and a short mathematical explanation.

## License

- **Code:** MIT License (see `LICENSE`).
- **Paper PDF:** includes its own license statement inside the document (CC BY 4.0 for the paper content, with exclusions for third-party material as noted in the PDF).
