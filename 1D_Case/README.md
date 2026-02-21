# 1D simulation (classical vs interval-extended)

This folder contains 1D simulations illustrating the effect of **unknown-but-bounded** remaining systematic error on congruency testing.

## Model (1D)
\[
d = \mu + b + e, \quad e\sim\mathcal N(0,\sigma^2), \quad b\in[-\Delta,\Delta].
\]

- Classical statistic: \(T_{\mathrm{cls}} = d^2/\sigma^2\)
- Interval-extended statistic: \(T_{\mathrm{ext}}(b) = (d-b)^2/\sigma^2\)
- Interval endpoints: \([T_{\min},T_{\max}]\) over \(b\in[-\Delta,\Delta]\)

Decision rule:
- strict accept if \(T_{\max}\le k_\alpha\)
- reject if \(T_{\min}\ge k_\alpha\)
- ambiguous otherwise

## Scripts
- `01_classic_1D.py` — classical test simulation
- `02_extended_box_1D.py` — interval-extended test with 1D box bias

Outputs are written to `outputs/`.
