# 1D simulations (classical vs interval-extended)

This folder contains 1D simulations illustrating the effect of **unknown-but-bounded** remaining systematic error on congruency testing.

## Model (1D)

$$
d = \mu + b + e,
\qquad
e\sim\mathcal N(0,\sigma^2),
\qquad
b\in B=[-\Delta,\Delta].
$$

- Classical statistic:

$$
T_{\mathrm{cls}} = \frac{d^2}{\sigma^2}.
$$

- Interval-extended statistic (for an admissible bias value \(b\)):

$$
T_{\mathrm{ext}}(b) = \frac{(d-b)^2}{\sigma^2},
\qquad b\in B.
$$

- Interval endpoints:

$$
[T]=[T_{\min},T_{\max}],
\qquad
T_{\min}=\min_{b\in B}T_{\mathrm{ext}}(b),
\quad
T_{\max}=\max_{b\in B}T_{\mathrm{ext}}(b).
$$

Decision rule:
- strict accept if $\(T_{\max}\le k_\alpha\)$
- reject if $\(T_{\min}\ge k_\alpha\)$
- ambiguous otherwise

## Scripts

- `01_classic_1D.py` — classical test simulation  
- `02_extended_box_1D.py` — interval-extended test with 1D box bias  

Outputs are written to `outputs/`.
