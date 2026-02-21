# 2D simulation (classical vs interval-extended)

This folder contains 2D simulations for planar displacement vectors and demonstrates the geometric interpretation of the interval-extended test.

## Model (2D)
\[
\mathbf d = \boldsymbol\mu_d + \mathbf b + \mathbf e, \quad \mathbf e\sim\mathcal N(0,\Sigma_d), \quad \mathbf b\in B\subset\mathbb R^2.
\]

- Classical statistic: \(T_{\mathrm{cls}} = \mathbf d^\top\Sigma_d^{-1}\mathbf d\)
- Classical acceptance ellipse: \(E = \{\mathbf d: \mathbf d^\top\Sigma_d^{-1}\mathbf d\le k_\alpha\}\)
- Interval-extended statistic: \(T_{\mathrm{ext}}(\mathbf b)=(\mathbf d-\mathbf b)^\top\Sigma_d^{-1}(\mathbf d-\mathbf b)\)
- Interval endpoints: \([T_{\min},T_{\max}]\) over \(\mathbf b\in B\)

Bias set models:
- **Box**: axis-aligned bounds
- **Zonotope**: generator-based bounds (preferred directions / dependencies)

Geometric view:
- \(A_{\mathrm{ext}} = E\oplus B\), \(A_{\mathrm{in}} = E\ominus B\), \(A_{\mathrm{amb}} = A_{\mathrm{ext}}\setminus A_{\mathrm{in}}\)

## Scripts
- `01_classic_2D.py`
- `02_extended_box_2D.py`
- `03_extended_zonotope_2D.py`

Outputs are written to `outputs/`. See `docs/minkowski/` for additional geometric intuition.
