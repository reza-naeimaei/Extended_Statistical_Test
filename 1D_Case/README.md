# 1D simulations (classical vs interval-extended)

This folder contains 1D simulations illustrating how **unknown-but-bounded** remaining systematic effects (bias) change the classical 1D congruency decision and lead to a three-valued outcome (accept / reject / ambiguous).

---

## Mathematical background (1D)

### Classical 1D congruency test

For a scalar displacement observation:

$$
d = \mu + e,
\qquad
e \sim \mathcal N(0,\sigma_d^2),
$$

we test

$$
H_0: \mu = 0
\quad\text{vs.}\quad
H_a: \mu \ne 0
$$

using the standardized statistic

$$
T_{\mathrm{cls}} = \frac{d}{\sigma_d},
\qquad
\text{reject if } |T_{\mathrm{cls}}| > k_\alpha,
\qquad
k_\alpha = \Phi^{-1}(1-\alpha/2).
$$

### Extension with bounded bias

Model remaining systematics as an unknown-but-bounded bias

$$
b \in [-\Delta_d,\Delta_d],
$$

and use

$$
d = \mu + b + e.
$$

The interval-extended statistic becomes

$$
T_{\mathrm{ext}} = [T_{\min},T_{\max}]
= \left[\frac{d-\Delta_d}{\sigma_d},\;\frac{d+\Delta_d}{\sigma_d}\right].
$$

With the classical acceptance interval

$$
A=[-k_\alpha,k_\alpha],
$$

the three-valued decision rule is:

- **accept** if $$T_{\mathrm{ext}} \subset A$$
- **reject** if $$T_{\mathrm{ext}} \cap A = \varnothing$$
- **ambiguous** otherwise

---

## Figures

**Equation summary (as used in the project)**

![Classical 1D and interval-extension equations](docs/1D_equations.png)

**Decision intuition (accept / ambiguous / reject regions)**

![Interval-extended 1D decision intuition](docs/1D_decision_rule.png)

---

## Scripts

- `01_classic_1D.py` — classical 1D test simulation  
- `02_extended_box_1D.py` — interval-extended 1D test with bounded bias  

Outputs are written to `outputs/`.
