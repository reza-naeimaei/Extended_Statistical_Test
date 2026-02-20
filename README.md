# Simulation Study (1D/2D) — Companion Code

## Description

This repository provides the complete simulation framework used in the study
presented in `Paper/ivk_2026_013.pdf`. 

It includes:

- Implementation of classical and extended methods in 1D and 2D
- Numerical experiments used for validation
- Generated figures and result files
- Reproducible scripts corresponding directly to the paper

The repository is structured to allow full reproducibility of the simulation
results reported in the publication.

---

## Repository Structure

- `1D/` — 1D simulation scripts and outputs  
- `2D/` — 2D simulation scripts and outputs  
- `Paper/` — Final published paper PDF  
- `*/outputs/` — Generated figures and saved result files  

---

## How to Run

### 1D Simulations
```
python 1D/01_classic_1D.py
python 1D/02_extended_box_1D.py
```

### 2D Simulations
```
python 2D/01_classic_2D.py
python 2D/02_extended_box_2D.py
python 2D/03_extended_zonotope_2D.py
```

Outputs are written to the corresponding `outputs/` folders.

---

## Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Then execute the desired simulation script.
