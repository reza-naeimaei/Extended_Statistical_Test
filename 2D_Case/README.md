# 2D simulations (classical vs interval-extended)

This folder contains 2D simulations for planar displacement vectors and demonstrates the geometric interpretation of the interval-extended test.

## Model (2D)

$$
\mathbf d = \boldsymbol\mu_d + \mathbf b + \mathbf e,
\qquad
\mathbf e\sim\mathcal N(\mathbf 0,\Sigma_d),
\qquad
\mathbf b\in B\subset\mathbb R^2.
$$

Hypotheses:

$$
H_0: \boldsymbol{\mu}_d = \mathbf 0
\quad\text{vs.}\quad
H_a: \boldsymbol{\mu}_d \ne \mathbf 0 .
$$

## Admissible bias set \(B\)

In this repository we use \(B\) as a generic symbol for the admissible set of remaining systematic effects, instantiated as:

**Box model**

$$
B_{\mathrm{box}} = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \subset \mathbb{R}^2.
$$

**Zonotope model**

$$
B_{\mathrm{zono}} = \{G\zeta : \zeta \in [-1,1]^p\}
= \{\sum_{i=1}^{p}\zeta_i g^{(i)} : \zeta_i \in [-1,1]\}.
$$

In the experiments, we evaluate the test for $\(B=B_{\mathrm{box}}\)$ and for $\(B=B_{\mathrm{zono}}\)$.

## Test statistics

**Classical statistic**

$$
T_{\mathrm{cls}} = \mathbf d^\top\Sigma_d^{-1}\mathbf d.
$$

**Classical acceptance ellipse**

$$
E = \{\mathbf d : \mathbf d^\top\Sigma_d^{-1}\mathbf d \le k_\alpha\}.
$$

**Interval-extended statistic**

$$
T_{\mathrm{ext}}(\mathbf b)=(\mathbf d-\mathbf b)^\top\Sigma_d^{-1}(\mathbf d-\mathbf b),
\qquad \mathbf b\in B.
$$

**Interval endpoints**

$$
[T]=[T_{\min},T_{\max}],
\qquad
T_{\min}=\min_{\mathbf b\in B}T_{\mathrm{ext}}(\mathbf b),
\quad
T_{\max}=\max_{\mathbf b\in B}T_{\mathrm{ext}}(\mathbf b).
$$

### Decision regions (geometric characterization)

| Region | Set expression | Test | Meaning |
|---|---|---|---|
| Outer region | $A_{\mathrm{ext}} = B \oplus E$ | $T_{\min} \ge k_{\alpha}$ | strict rejection |
| Inner region | $A_{\mathrm{in}} = E \ominus B$ | $T_{\max} \le k_{\alpha}$ | strict acceptance |
| Ambiguous region | $A_{\mathrm{amb}} = A_{\mathrm{ext}} \setminus A_{\mathrm{in}}$ | $T_{\min} \le k_{\alpha} < T_{\max}$ | not separable |}

## Geometric view (Minkowski operations)

See [docs/](../docs/) for additional geometric intuition and animations.

## Scripts

- `01_classic_2D.py`
- `02_extended_box_2D.py`
- `03_extended_zonotope_2D.py`

Outputs are written to `outputs/`.
