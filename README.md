# RANSAC vs. Regularised & Robust Regression

A from-scratch comparison of **seven regression methods** on data that's been deliberately polluted with **40% outliers**. The headline question: *which methods actually survive bad data, and which ones only pretend to?*

The short answer — regularisation (Ridge, Lasso, ElasticNet) does **not** make regression robust to outliers. Only methods that explicitly down-weight or reject bad points (Huber, Theil-Sen, RANSAC) recover the true line.

---

## The Problem in One Picture

We generate 200 points along a known line, `y = 1.5x + 2.0`:

- **60% are inliers** — points sitting on the line with a little Gaussian noise.
- **40% are outliers** — points with random `y` values, scattered anywhere.

Then we fit all seven methods to this same messy dataset and check how close each one gets back to the *true* slope (1.5) and intercept (2.0).

Because we know which points are the "real" ones, we can score each method fairly — an **oracle test** — by measuring its error on the clean inliers only.

---

## The Seven Methods

| Method | What it does | Robust to outliers? | Breakdown point\* |
|---|---|:---:|:---:|
| **OLS** | Plain least squares — minimises squared error |  No | 0% |
| **Ridge** | OLS + L2 penalty (shrinks coefficients) |  No | 0% |
| **Lasso** | OLS + L1 penalty (sparse coefficients) |  No | 0% |
| **ElasticNet** | OLS + L1 + L2 combined |  No | 0% |
| **Huber** | Switches from squared to linear loss for big errors |  Partial | ~50% |
| **Theil-Sen** | Median of slopes between all point pairs |  Yes | ~29.3% |
| **RANSAC** | Votes on the line with the most agreeing points |  Yes | up to ~99% |

\* *Breakdown point = the fraction of garbage data the method can tolerate before its answer becomes meaningless.*

### Why regularisation doesn't help

This is the core lesson. Ridge, Lasso, and ElasticNet all add a *penalty on the coefficients*, but they leave the **data term untouched** — it's still the sum of squared residuals:

```
Loss = Σ (yᵢ - ŷᵢ)²  +  penalty(β)
       └──────────────┘
        unchanged — this is where outliers do their damage
```

A single outlier sitting `10σ` away from the line contributes `100σ²` to the loss. That one point can outweigh dozens of honest points combined, so the fit gets dragged toward it. Adding a penalty term does nothing to stop this — penalties control *model complexity*, not *outlier influence*. That's why Ridge/Lasso/ElasticNet score almost identically to plain OLS here.

To be robust, you have to change how individual residuals are treated:

- **Huber** caps the growth of large residuals — beyond a threshold `δ`, error grows *linearly* instead of *quadratically*, so outliers pull less. (Soft down-weighting, not rejection.)
- **Theil-Sen** uses the **median** of pairwise slopes. Medians ignore extremes by construction, so up to ~29% of the data can be junk and the answer barely moves.
- **RANSAC** goes further and *rejects* outliers outright (explained below).

---

## How RANSAC Works

RANSAC (**RAN**dom **SA**mple **C**onsensus) doesn't try to fit all the data at once. Instead it repeatedly guesses a line from a tiny random sample, then asks *"how many points agree with this guess?"* The guess with the most agreement wins.

The loop, step by step:

1. **Sample** — pick 2 random points (the minimum needed to define a line).
2. **Hypothesise** — draw the exact line through those 2 points.
3. **Consensus** — count how many of *all* points fall within a distance `τ` of that line. These are the "inliers" that vote for this hypothesis.
4. **Evaluate** — if this line has more inliers than any previous one, keep it.
5. **Adapt** — re-estimate how many more iterations are needed (see formula below) and stop early once we're confident.
6. **Refine** — once the best line is found, refit it properly using least squares on *only* its inliers.

The point-to-line distance used for the inlier test is the perpendicular distance:

```
distance = |m·x − y + c| / √(m² + 1)
```

### How many iterations do we need?

We don't pick the iteration count by guessing. There's a formula. If we want probability `p` of hitting at least one all-inlier sample, with outlier ratio `ε` and sample size `s`:

```
        log(1 − p)
N  ≥  ──────────────────
       log(1 − (1−ε)ˢ)
```

The implementation uses this **adaptively** — every time it finds a better line, it re-estimates `ε` from the current inlier count and recalculates `N`. As the estimate of the data quality improves, the required number of iterations usually drops, so it stops early instead of burning all 500 iterations.

For `s = 2` and a 40% outlier rate, this needs only a handful of iterations to be 99% confident — which is exactly what makes RANSAC practical.

---

## What's in This Repo

| File | What it is |
|---|---|
| `01_ransac_scratch.ipynb` | The full notebook — theory in markdown, all seven methods implemented from scratch, comparison tables, and plots. Start here. |
| `01a_all_methods_lines.png` | Scatter plot with all seven fitted lines, the RANSAC iteration history, and per-method residual distributions. |
| `01b_metrics_comparison.png` | Bar charts of slope error, intercept error, and inlier MSE for each method. |

Everything is built with just **NumPy and Matplotlib** — no scikit-learn. Ridge has a closed-form solution; Lasso and ElasticNet use coordinate descent with soft-thresholding; Huber uses iteratively re-weighted least squares (IRLS); Theil-Sen and RANSAC are implemented directly from their definitions. The goal is to see exactly what each method does rather than call a black box.

---

## Running It

You'll need Python with a few standard packages:

```bash
pip install numpy matplotlib
```

Then open the notebook:

```bash
jupyter notebook 01_ransac_scratch.ipynb
```

Run the cells top to bottom. It prints the comparison tables, saves the two figures next to the notebook, and ends with a summary of the best- and worst-performing methods.

---

## The Takeaway

> Regularisation controls **how complex** your model is.
> Robustness controls **how much bad data can hurt you.**
> They are not the same thing — and on contaminated data, only robustness matters.

If your data might contain outliers, reach for Huber, Theil-Sen, or RANSAC. Reaching for Ridge or Lasso and expecting them to clean up outliers is a common and costly mistake.
