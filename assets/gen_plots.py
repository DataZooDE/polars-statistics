"""Generate all static PNG plots for the documentation examples.

Run from the examples/ directory:
    uv run python ../docs/assets/gen_plots.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = "seaborn-v0_8-whitegrid"
DPI = 150
SAVE_KW = {"dpi": DPI, "bbox_inches": "tight", "facecolor": "white"}


def set_style():
    plt.style.use(STYLE)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
    })


# ── 1. hyp_distributions.png ────────────────────────────────────────────────
def plot_hyp_distributions():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    drug = [12.3, 10.1, 14.5, 11.8, 13.2, 9.7, 15.1, 12.8, 10.5, 14.0,
            11.2, 13.9, 10.8, 12.6, 14.3, 11.5, 13.7, 10.2, 12.1, 15.0]
    placebo = [8.1, 7.5, 9.2, 6.8, 10.1, 7.9, 8.5, 9.8, 7.2, 8.7,
               9.5, 6.5, 8.3, 7.1, 9.9, 8.0, 7.6, 9.3, 6.9, 8.8]

    bins = np.linspace(5, 17, 13)
    ax.hist(drug, bins=bins, alpha=0.6, label="Drug", color="#4C72B0", edgecolor="white")
    ax.hist(placebo, bins=bins, alpha=0.6, label="Placebo", color="#DD8452", edgecolor="white")
    ax.axvline(np.mean(drug), color="#4C72B0", ls="--", lw=1.5, label=f"Drug mean ({np.mean(drug):.1f})")
    ax.axvline(np.mean(placebo), color="#DD8452", ls="--", lw=1.5, label=f"Placebo mean ({np.mean(placebo):.1f})")
    ax.set_xlabel("Blood Pressure Reduction (mmHg)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Blood Pressure Reduction")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "hyp_distributions.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ hyp_distributions.png")


# ── 2. hyp_correlation.png ──────────────────────────────────────────────────
def plot_hyp_correlation():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    temp = np.array([20, 22, 25, 27, 30, 32, 35, 37, 40, 42], dtype=float)
    sales = np.array([200, 220, 280, 300, 350, 380, 420, 440, 500, 520], dtype=float)

    ax.scatter(temp, sales, s=60, zorder=5, color="#4C72B0", edgecolor="white", lw=0.8)
    m, b = np.polyfit(temp, sales, 1)
    x_line = np.linspace(18, 44, 100)
    ax.plot(x_line, m * x_line + b, color="#C44E52", lw=2,
            label=f"y = {m:.1f}x {b:+.0f}  (r = 0.999)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Ice Cream Sales")
    ax.set_title("Temperature vs Ice Cream Sales")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "hyp_correlation.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ hyp_correlation.png")


# ── 3. reg_actual_vs_pred.png ───────────────────────────────────────────────
def plot_reg_actual_vs_pred():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    actual = np.array([250, 320, 280, 450, 380, 520, 290, 410, 350, 600,
                       270, 340, 310, 480, 400, 550, 300, 430, 370, 620], dtype=float)
    predicted = np.array([253.70, 319.13, 280.13, 456.69, 387.90, 527.26, 264.81,
                          424.14, 359.95, 587.85, 254.16, 328.71, 304.49, 465.65,
                          396.87, 518.30, 261.69, 434.49, 368.91, 601.91])
    lower = np.array([224.10, 288.82, 248.74, 427.59, 358.88, 495.36, 235.10,
                      394.89, 330.13, 554.66, 222.37, 298.36, 274.64, 436.23,
                      367.65, 487.66, 231.40, 405.16, 339.02, 569.17])
    upper = np.array([283.29, 349.43, 311.52, 485.79, 416.92, 559.16, 294.52,
                      453.39, 389.77, 621.03, 285.94, 359.06, 334.34, 495.08,
                      426.08, 548.95, 291.98, 463.82, 398.80, 634.66])

    order = np.argsort(actual)
    actual_s, predicted_s = actual[order], predicted[order]
    lower_s, upper_s = lower[order], upper[order]

    ax.fill_between(actual_s, lower_s, upper_s, alpha=0.2, color="#4C72B0",
                    label="95% Prediction Interval")
    ax.scatter(actual, predicted, s=50, zorder=5, color="#4C72B0", edgecolor="white", lw=0.8)
    lim = [220, 650]
    ax.plot(lim, lim, ls="--", color="#999999", lw=1, label="Perfect fit")
    ax.set_xlabel("Actual Price ($k)")
    ax.set_ylabel("Predicted Price ($k)")
    ax.set_title("Actual vs Predicted (OLS)")
    ax.legend(frameon=True, fancybox=True, loc="upper left")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.savefig(OUT / "reg_actual_vs_pred.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_actual_vs_pred.png")


# ── 4. reg_residuals.png ────────────────────────────────────────────────────
def plot_reg_residuals():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    predicted = np.array([253.70, 319.13, 280.13, 456.69, 387.90, 527.26, 264.81,
                          424.14, 359.95, 587.85, 254.16, 328.71, 304.49, 465.65,
                          396.87, 518.30, 261.69, 434.49, 368.91, 601.91])
    actual = np.array([250, 320, 280, 450, 380, 520, 290, 410, 350, 600,
                       270, 340, 310, 480, 400, 550, 300, 430, 370, 620], dtype=float)
    residuals = actual - predicted

    ax.scatter(predicted, residuals, s=50, color="#4C72B0", edgecolor="white", lw=0.8, zorder=5)
    ax.axhline(0, color="#C44E52", ls="--", lw=1.5)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    fig.savefig(OUT / "reg_residuals.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_residuals.png")


# ── 5. reg_quantile_lines.png ───────────────────────────────────────────────
def plot_reg_quantile_lines():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    sqft = np.array([800, 1000, 850, 1400, 1200, 1600, 900, 1300, 1100, 1800,
                     820, 1050, 950, 1450, 1250, 1550, 880, 1350, 1150, 1900], dtype=float)
    price = np.array([250, 320, 280, 450, 380, 520, 290, 410, 350, 600,
                      270, 340, 310, 480, 400, 550, 300, 430, 370, 620], dtype=float)

    ax.scatter(sqft, price, s=50, color="#666666", edgecolor="white", lw=0.8, zorder=5, label="Data")

    x = np.linspace(750, 1950, 100)
    # Simplified quantile lines (intercept + sqft coeff only, dominant predictor)
    # Q25: int=-60.00, sqft=0.34
    ax.plot(x, -60.00 + 0.34 * x, color="#4C72B0", lw=2, ls="--", label="τ = 0.25")
    # Q50: int=-58.38, sqft=0.34
    ax.plot(x, -58.38 + 0.34 * x, color="#C44E52", lw=2, label="τ = 0.50 (median)")
    # Q75: int=-57.58, sqft=0.38
    ax.plot(x, -57.58 + 0.38 * x, color="#55A868", lw=2, ls="-.", label="τ = 0.75")

    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Price ($k)")
    ax.set_title("Quantile Regression (Price ~ Sqft)")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "reg_quantile_lines.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_quantile_lines.png")


# ── 6. grp_r2_bars.png ─────────────────────────────────────────────────────
def plot_grp_r2_bars():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))

    regions = ["North", "South", "West"]
    r2 = [0.9656, 0.9641, 0.9757]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars = ax.barh(regions, r2, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("R²")
    ax.set_title("Model Fit by Region (OLS)")
    ax.set_xlim(0.95, 0.985)
    for bar, v in zip(bars, r2):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    fig.savefig(OUT / "grp_r2_bars.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ grp_r2_bars.png")


# ── 7. grp_regression_lines.png ─────────────────────────────────────────────
def plot_grp_regression_lines():
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    # Data per region
    sales_all = [120, 135, 142, 128, 155, 138, 145, 132, 150, 140,
                 125, 148, 133, 156, 141, 130, 152, 137, 144, 127,
                 158, 136, 149, 131, 153, 139, 146, 134, 151, 143,
                 200, 215, 225, 208, 235, 218, 228, 212, 232, 220,
                 205, 230, 213, 238, 222, 210, 234, 217, 226, 207,
                 240, 216, 231, 211, 236, 219, 227, 214, 233, 223,
                 160, 175, 168, 182, 170, 190, 165, 185, 172, 178,
                 163, 188, 171, 193, 176, 167, 191, 174, 183, 162,
                 195, 173, 186, 164, 192, 177, 184, 169, 189, 180]
    price_all = [25, 28, 30, 26, 32, 29, 31, 27, 33, 28,
                 24, 31, 27, 34, 29, 26, 33, 28, 30, 25,
                 35, 27, 32, 26, 34, 29, 31, 27, 33, 30,
                 40, 43, 45, 41, 47, 44, 46, 42, 48, 43,
                 39, 46, 42, 49, 44, 41, 47, 43, 45, 40,
                 50, 44, 47, 42, 48, 44, 46, 43, 48, 45,
                 30, 33, 32, 35, 31, 37, 30, 36, 33, 34,
                 29, 36, 32, 38, 34, 31, 37, 33, 35, 29,
                 39, 33, 36, 30, 38, 34, 35, 32, 37, 34]

    regions = {"North": (0, 30), "South": (30, 60), "West": (60, 90)}
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (name, (lo, hi)), color in zip(axes, regions.items(), colors):
        s = np.array(sales_all[lo:hi], dtype=float)
        p = np.array(price_all[lo:hi], dtype=float)
        ax.scatter(p, s, s=30, color=color, edgecolor="white", lw=0.6, alpha=0.8)
        m, b = np.polyfit(p, s, 1)
        x = np.linspace(p.min() - 1, p.max() + 1, 50)
        ax.plot(x, m * x + b, color=color, lw=2)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Price")
        if lo == 0:
            ax.set_ylabel("Sales")

    fig.suptitle("Sales ~ Price by Region", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "grp_regression_lines.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ grp_regression_lines.png")


# ── 8. abt_conversion_rates.png ─────────────────────────────────────────────
def plot_abt_conversion_rates():
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    groups = ["Control", "Treatment"]
    rates = [12.0, 14.5]
    ci_lower = [10.0, 12.3]
    ci_upper = [14.0, 16.7]
    errors = [[r - lo for r, lo in zip(rates, ci_lower)],
              [hi - r for r, hi in zip(rates, ci_upper)]]
    colors = ["#4C72B0", "#55A868"]

    bars = ax.bar(groups, rates, color=colors, edgecolor="white", width=0.5,
                  yerr=errors, capsize=8, error_kw={"lw": 1.5, "color": "#333333"})
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("A/B Test: Conversion Rates")
    ax.set_ylim(0, 20)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.02, "p = 0.099 (not significant at α = 0.05)",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic", color="#666666")
    fig.savefig(OUT / "abt_conversion_rates.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ abt_conversion_rates.png")


# ── 9. abt_tost_diagram.png ─────────────────────────────────────────────────
def plot_abt_tost_diagram():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 2.5))

    delta = 0.5
    estimate = 0.095
    ci_lo = -0.1213
    ci_hi = 0.3113

    # Equivalence bounds
    ax.axvline(-delta, color="#C44E52", ls="--", lw=2, label=f"Equivalence bounds (±{delta})")
    ax.axvline(delta, color="#C44E52", ls="--", lw=2)
    ax.axvspan(-delta, delta, alpha=0.08, color="#55A868")

    # CI
    ax.plot([ci_lo, ci_hi], [0.5, 0.5], color="#4C72B0", lw=3, solid_capstyle="round")
    ax.plot(estimate, 0.5, "o", color="#4C72B0", ms=10, zorder=5)

    # Labels
    ax.text(-delta, 1.05, f"−{delta}", ha="center", fontsize=10, color="#C44E52")
    ax.text(delta, 1.05, f"+{delta}", ha="center", fontsize=10, color="#C44E52")
    ax.text(estimate, 0.15, f"Δ = {estimate:.3f}", ha="center", fontsize=10, color="#4C72B0")
    ax.text(0, 1.4, "Equivalence Region", ha="center", fontsize=10, color="#55A868", fontweight="bold")
    ax.text(0, -0.6, "CI entirely inside bounds → Equivalent", ha="center",
            fontsize=9, style="italic", color="#666666")

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("Mean Difference")
    ax.set_title("TOST Equivalence Test")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "abt_tost_diagram.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ abt_tost_diagram.png")


# ── 10. hyp_boxplot_multigroup.png ─────────────────────────────────────────
def plot_hyp_boxplot_multigroup():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    method_a = [85, 90, 88, 92, 87, 91, 86, 89, 93, 88]
    method_b = [78, 82, 80, 84, 79, 83, 77, 81, 85, 80]
    method_c = [72, 76, 74, 78, 73, 77, 71, 75, 79, 74]

    bp = ax.boxplot(
        [method_a, method_b, method_c],
        tick_labels=["Method A", "Method B", "Method C"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#333333", "lw": 2},
    )
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (data, color) in enumerate(zip([method_a, method_b, method_c], colors)):
        jitter = np.random.default_rng(42).normal(0, 0.04, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   s=25, color=color, edgecolor="white", lw=0.6, zorder=5, alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Multi-Group Comparison (Kruskal-Wallis)")
    ax.text(0.5, -0.12, "H = 24.60, p < 0.001 — at least one group differs",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic", color="#666666")
    fig.savefig(OUT / "hyp_boxplot_multigroup.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ hyp_boxplot_multigroup.png")


# ── 11. reg_coef_forest.png ──────────────────────────────────────────────
def plot_reg_coef_forest():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    terms = ["sqft (x1)", "bedrooms (x2)", "age (x3)"]
    estimates = [0.3607, 1.7018, 1.6806]
    std_errors = [0.0332, 11.4411, 1.3684]
    ci_lower = [e - 1.96 * se for e, se in zip(estimates, std_errors)]
    ci_upper = [e + 1.96 * se for e, se in zip(estimates, std_errors)]

    y_pos = np.arange(len(terms))
    ax.errorbar(estimates, y_pos, xerr=[[e - lo for e, lo in zip(estimates, ci_lower)],
                                         [hi - e for e, hi in zip(estimates, ci_upper)]],
                fmt="o", color="#4C72B0", ms=8, capsize=6, lw=2, capthick=1.5,
                ecolor="#4C72B0", markeredgecolor="white", markeredgewidth=1)
    ax.axvline(0, color="#C44E52", ls="--", lw=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms)
    ax.set_xlabel("Coefficient Estimate (± 95% CI)")
    ax.set_title("OLS Coefficient Forest Plot")
    ax.invert_yaxis()
    fig.savefig(OUT / "reg_coef_forest.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_coef_forest.png")


# ── 12. reg_ci_vs_pi.png ────────────────────────────────────────────────
def plot_reg_ci_vs_pi():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    actual = np.array([250, 270, 280, 290, 300, 310, 320, 340, 350, 370,
                       380, 400, 410, 430, 450, 480, 520, 550, 600, 620], dtype=float)
    predicted = np.array([253.70, 254.16, 280.13, 264.81, 304.49, 261.69,
                          319.13, 328.71, 359.95, 368.91,
                          387.90, 396.87, 424.14, 434.49, 456.69, 465.65,
                          527.26, 518.30, 587.85, 601.91])
    # CI (confidence interval for mean)
    ci_lo = np.array([242.63, 222.37, 264.90, 235.10, 274.64, 231.40,
                      306.29, 298.36, 330.13, 339.02,
                      378.48, 367.65, 394.89, 405.16, 447.03, 436.23,
                      495.36, 487.66, 554.66, 569.17])
    ci_hi = np.array([264.76, 285.94, 295.36, 294.52, 334.34, 291.98,
                      331.97, 359.06, 389.77, 398.80,
                      397.31, 426.08, 453.39, 463.82, 466.35, 495.08,
                      559.16, 548.95, 621.03, 634.66])
    # PI (prediction interval for individual)
    pi_lo = np.array([224.10, 222.37, 248.74, 235.10, 274.64, 231.40,
                      288.82, 298.36, 330.13, 339.02,
                      358.88, 367.65, 394.89, 405.16, 427.59, 436.23,
                      495.36, 487.66, 554.66, 569.17])
    pi_hi = np.array([283.29, 285.94, 311.52, 294.52, 334.34, 291.98,
                      349.43, 359.06, 389.77, 398.80,
                      416.92, 426.08, 453.39, 463.82, 485.79, 495.08,
                      559.16, 548.95, 621.03, 634.66])

    # Use sorted actual for band plotting
    order = np.argsort(actual)
    a_s = actual[order]
    p_s, ci_lo_s, ci_hi_s = predicted[order], ci_lo[order], ci_hi[order]
    pi_lo_s, pi_hi_s = pi_lo[order], pi_hi[order]

    ax.fill_between(a_s, pi_lo_s, pi_hi_s, alpha=0.15, color="#DD8452",
                    label="95% Prediction Interval")
    ax.fill_between(a_s, ci_lo_s, ci_hi_s, alpha=0.3, color="#4C72B0",
                    label="95% Confidence Interval")
    ax.scatter(actual, predicted, s=50, color="#4C72B0", edgecolor="white",
               lw=0.8, zorder=5)
    lim = [220, 650]
    ax.plot(lim, lim, ls="--", color="#999999", lw=1, label="Perfect fit")
    ax.set_xlabel("Actual Price ($k)")
    ax.set_ylabel("Predicted Price ($k)")
    ax.set_title("Confidence Interval vs Prediction Interval")
    ax.legend(frameon=True, fancybox=True, loc="upper left")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.savefig(OUT / "reg_ci_vs_pi.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_ci_vs_pi.png")


# ── 13. reg_regularization_coefs.png ─────────────────────────────────────
def plot_reg_regularization_coefs():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    terms = ["sqft", "bedrooms", "age"]
    ols_coefs = [0.3607, 1.7018, 1.6806]
    ridge_coefs = [0.3620, 0.9253, 1.6313]
    enet_coefs = [0.0, 0.7043, 1.1512]

    x = np.arange(len(terms))
    w = 0.22
    ax.bar(x - w, ols_coefs, w, label="OLS", color="#4C72B0", edgecolor="white")
    ax.bar(x, ridge_coefs, w, label="Ridge (λ=1)", color="#DD8452", edgecolor="white")
    ax.bar(x + w, enet_coefs, w, label="ElasticNet (λ=1, α=0.5)", color="#55A868",
           edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Regularization Effect on Coefficients")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    ax.axhline(0, color="#333333", lw=0.8)
    fig.savefig(OUT / "reg_regularization_coefs.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg_regularization_coefs.png")


# ── 14. grp_paired_before_after.png ──────────────────────────────────────
def plot_grp_paired_before_after():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    groups = {
        "Group A": {
            "before": [85, 90, 78, 92, 88, 76, 95, 83, 89, 91,
                       80, 87, 93, 84, 86, 79, 94, 82, 88, 90],
            "after":  [89, 94, 83, 96, 92, 81, 99, 88, 93, 95,
                       85, 91, 97, 89, 90, 84, 98, 87, 92, 94],
        },
        "Group B": {
            "before": [70, 75, 68, 77, 73, 66, 80, 72, 74, 76,
                       65, 72, 78, 69, 71, 64, 79, 67, 73, 75],
            "after":  [72, 78, 70, 80, 76, 69, 83, 75, 77, 79,
                       68, 75, 81, 72, 74, 67, 82, 70, 76, 78],
        },
    }

    colors = ["#4C72B0", "#DD8452"]
    for ax, (name, data), color in zip(axes, groups.items(), colors):
        before = np.array(data["before"])
        after = np.array(data["after"])
        for b, a in zip(before, after):
            ax.plot([0, 1], [b, a], color=color, alpha=0.35, lw=1)
        ax.scatter(np.zeros(len(before)), before, s=30, color=color, edgecolor="white",
                   lw=0.6, zorder=5, label="Before")
        ax.scatter(np.ones(len(after)), after, s=30, color=color, edgecolor="white",
                   lw=0.6, zorder=5, label="After")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"])
        ax.set_title(name, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, fontsize=9, loc="lower right")
        if name == "Group A":
            ax.set_ylabel("Score")

    fig.suptitle("Paired Before → After Scores", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "grp_paired_before_after.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ grp_paired_before_after.png")


# ── 15. grp_aic_comparison.png ───────────────────────────────────────────
def plot_grp_aic_comparison():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    regions = ["North", "South", "West"]
    aic_additive = [130.18, 136.16, 121.92]
    aic_interaction = [409.99, 422.10, 399.32]

    x = np.arange(len(regions))
    w = 0.3
    ax.bar(x - w / 2, aic_additive, w, label="Additive (price + ad)",
           color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, aic_interaction, w, label="Interaction (price × ad)",
           color="#DD8452", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("AIC (lower = better)")
    ax.set_title("Model Comparison: AIC by Region")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    fig.savefig(OUT / "grp_aic_comparison.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ grp_aic_comparison.png")


# ── 16. abt_revenue_distributions.png ────────────────────────────────────
def plot_abt_revenue_distributions():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    control = [12.5, 0, 8.3, 0, 15.2, 0, 22.1, 0, 9.8, 0,
               18.4, 0, 7.6, 0, 25.3, 0, 11.0, 0, 14.7, 0,
               0, 16.8, 0, 19.5, 0, 6.2, 0, 21.9, 0, 13.1]
    treatment = [15.8, 0, 11.2, 0, 18.5, 3.2, 28.4, 0, 13.1, 0,
                 22.7, 0, 10.9, 5.1, 30.2, 0, 14.3, 0, 17.6, 0,
                 2.8, 20.1, 0, 24.8, 0, 9.5, 0, 26.3, 3.7, 16.4]

    bins = np.linspace(0, 32, 17)
    ax.hist(control, bins=bins, alpha=0.6, label=f"Control (mean={np.mean(control):.1f})",
            color="#4C72B0", edgecolor="white")
    ax.hist(treatment, bins=bins, alpha=0.6, label=f"Treatment (mean={np.mean(treatment):.1f})",
            color="#55A868", edgecolor="white")
    ax.axvline(np.mean(control), color="#4C72B0", ls="--", lw=1.5)
    ax.axvline(np.mean(treatment), color="#55A868", ls="--", lw=1.5)
    ax.set_xlabel("Revenue per User ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Revenue Distribution: Control vs Treatment")
    ax.legend(frameon=True, fancybox=True)
    ax.text(0.5, -0.12, "Both groups non-normal (p < 0.001) — use non-parametric tests",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic", color="#666666")
    fig.savefig(OUT / "abt_revenue_distributions.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ abt_revenue_distributions.png")


# ── 17. abt_segment_forest.png ───────────────────────────────────────────
def plot_abt_segment_forest():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))

    segments = ["desktop", "mobile", "tablet"]
    estimates = [0.2000, 0.5100, 0.2000]
    ci_lower = [-0.0400, 0.2188, 0.0136]
    ci_upper = [0.4400, 0.8012, 0.3864]

    y_pos = np.arange(len(segments))
    colors = ["#4C72B0" if lo > 0 else "#999999" for lo in ci_lower]

    for i, (est, lo, hi, color) in enumerate(zip(estimates, ci_lower, ci_upper, colors)):
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]],
                    fmt="o", color=color, ms=8, capsize=6, lw=2, capthick=1.5,
                    ecolor=color, markeredgecolor="white", markeredgewidth=1)

    ax.axvline(0, color="#C44E52", ls="--", lw=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(segments)
    ax.set_xlabel("Mean Difference (Treatment − Control)")
    ax.set_title("Per-Segment Treatment Effect")
    ax.invert_yaxis()

    # Significance markers
    for i, (lo, p) in enumerate(zip(ci_lower, [0.168, 0.005, 0.078])):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(ci_upper[i] + 0.03, i, sig, va="center", fontsize=10,
                color="#333333" if sig != "ns" else "#999999")

    fig.savefig(OUT / "abt_segment_forest.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ abt_segment_forest.png")


# ── 18. glm_poisson_fit.png ────────────────────────────────────────────────
def plot_glm_poisson_fit():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    age = np.array([22, 24, 25, 23, 26, 27, 28, 24, 29, 30,
                    35, 38, 40, 36, 42, 44, 39, 45, 41, 46,
                    52, 55, 58, 54, 60, 56, 62, 64, 66, 68], dtype=float)
    claims = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                       1, 1, 2, 1, 1, 2, 1, 2, 1, 2,
                       2, 2, 3, 2, 3, 2, 3, 3, 4, 4], dtype=float)

    ax.scatter(age, claims, s=60, color="#4C72B0", edgecolor="white", lw=0.8,
               zorder=5, label="Observed claims")

    # Poisson fitted curve: intercept=-1.8421, coef=[0.0380]
    x_fit = np.linspace(20, 70, 200)
    y_fit = np.exp(-1.8421 + 0.0380 * x_fit)
    ax.plot(x_fit, y_fit, color="#C44E52", lw=2.5, label="Poisson fit (exp link)")

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Poisson Regression: Age → Claims")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "glm_poisson_fit.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ glm_poisson_fit.png")


# ── 19. glm_deviance_residuals.png ────────────────────────────────────────
def plot_glm_deviance_residuals():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    age = np.array([22, 24, 25, 23, 26, 27, 28, 24, 29, 30,
                    35, 38, 40, 36, 42, 44, 39, 45, 41, 46,
                    52, 55, 58, 54, 60, 56, 62, 64, 66, 68], dtype=float)
    claims = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                       1, 1, 2, 1, 1, 2, 1, 2, 1, 2,
                       2, 2, 3, 2, 3, 2, 3, 3, 4, 4], dtype=float)
    fitted = np.exp(-1.8421 + 0.0380 * age)
    # Deviance residuals for Poisson
    sign = np.sign(claims - fitted)
    dev = np.where(claims == 0,
                   np.sqrt(2 * fitted),
                   np.sqrt(2 * (claims * np.log(claims / fitted) - (claims - fitted))))
    residuals = sign * dev

    ax.scatter(fitted, residuals, s=50, color="#4C72B0", edgecolor="white", lw=0.8, zorder=5)
    ax.axhline(0, color="#C44E52", ls="--", lw=1.5)
    ax.axhline(2, color="#999", ls=":", lw=1)
    ax.axhline(-2, color="#999", ls=":", lw=1)
    ax.set_xlabel("Fitted Values (μ)")
    ax.set_ylabel("Deviance Residual")
    ax.set_title("Poisson Deviance Residuals")
    fig.savefig(OUT / "glm_deviance_residuals.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ glm_deviance_residuals.png")


# ── 20. glm_link_comparison.png ───────────────────────────────────────────
def plot_glm_link_comparison():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Binary churn data: tenure as single predictor (simplified)
    x = np.linspace(-3, 3, 200)
    # Logit: 1/(1+exp(-x))
    logit_p = 1 / (1 + np.exp(-x))
    # Probit: Φ(x) — use erf approximation (no scipy needed)
    from math import erf, sqrt
    probit_p = np.array([0.5 * (1 + erf(xi / sqrt(2))) for xi in x])
    # Cloglog: 1 - exp(-exp(x))
    cloglog_p = 1 - np.exp(-np.exp(x))

    ax.plot(x, logit_p, lw=2.5, color="#4C72B0", label="Logit")
    ax.plot(x, probit_p, lw=2.5, color="#DD8452", ls="--", label="Probit")
    ax.plot(x, cloglog_p, lw=2.5, color="#55A868", ls="-.", label="Cloglog")

    ax.axhline(0.5, color="#999", ls=":", lw=1)
    ax.set_xlabel("Linear Predictor (η)")
    ax.set_ylabel("P(Y = 1)")
    ax.set_title("Link Function Comparison")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "glm_link_comparison.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ glm_link_comparison.png")


# ── 21. reg2_lasso_coefs.png ──────────────────────────────────────────────
def plot_reg2_lasso_coefs():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    terms = ["temp", "humidity", "occupancy", "sqft", "insulation"]
    # Lasso λ=0.1 coefficients from compute_regularized.py
    lasso_coefs = [-8.3988, 0.4017, -8.4682, 0.0, 5.0396]
    # OLS comparison (all 5 features)
    ols_coefs = [-8.7866, 0.3816, -8.2371, 0.002, 5.0775]

    x = np.arange(len(terms))
    w = 0.3
    ax.bar(x - w / 2, ols_coefs, w, label="OLS", color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, lasso_coefs, w, label="Lasso (λ=0.1)", color="#55A868", edgecolor="white")

    # Highlight zeroed coefficient
    ax.annotate("Zeroed by Lasso →", xy=(3, 0), xytext=(3, -3),
                fontsize=9, color="#C44E52", ha="center",
                arrowprops={"arrowstyle": "->", "color": "#C44E52"})

    ax.set_xticks(x)
    ax.set_xticklabels(terms)
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Lasso Feature Selection (λ = 0.1)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    fig.savefig(OUT / "reg2_lasso_coefs.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg2_lasso_coefs.png")


# ── 22. reg2_wls_residuals.png ────────────────────────────────────────────
def plot_reg2_wls_residuals():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Simulated data for OLS vs WLS residual comparison
    np.random.seed(42)
    sqft = np.linspace(800, 3000, 30)
    # Heteroskedastic noise: variance increases with sqft
    noise = np.random.normal(0, sqft / 300)
    kwh = 100 + 0.1 * sqft + noise

    # OLS residuals (unweighted)
    from numpy.polynomial import polynomial as P
    ols_c = P.polyfit(sqft, kwh, 1)
    ols_pred = P.polyval(sqft, ols_c)
    ols_resid = kwh - ols_pred

    # WLS residuals (weighted by 1/sqft)
    w = 1000 / sqft
    wls_c = P.polyfit(sqft, kwh, 1, w=np.sqrt(w))
    wls_pred = P.polyval(sqft, wls_c)
    wls_resid = (kwh - wls_pred) * np.sqrt(w)

    ax = axes[0]
    ax.scatter(sqft, ols_resid, s=40, color="#4C72B0", edgecolor="white", lw=0.6)
    ax.axhline(0, color="#C44E52", ls="--", lw=1.5)
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Residual")
    ax.set_title("OLS Residuals (heteroskedastic)")

    ax = axes[1]
    ax.scatter(sqft, wls_resid, s=40, color="#55A868", edgecolor="white", lw=0.6)
    ax.axhline(0, color="#C44E52", ls="--", lw=1.5)
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Weighted Residual")
    ax.set_title("WLS Residuals (corrected)")

    fig.tight_layout()
    fig.savefig(OUT / "reg2_wls_residuals.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg2_wls_residuals.png")


# ── 23. reg2_isotonic_fit.png ─────────────────────────────────────────────
def plot_reg2_isotonic_fit():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # sqft vs kwh (monotone increasing relationship)
    sqft = np.array([750, 780, 800, 850, 870, 880, 900, 950, 1000, 1050,
                     1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1550, 1600,
                     1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250,
                     2300, 2350, 2400, 2500, 2600, 2650, 2700, 2750, 2800, 2850,
                     2900, 2950, 3000], dtype=float)
    kwh = np.array([180, 176, 189, 184, 192, 195, 198, 210, 215, 220,
                    228, 223, 245, 234, 241, 252, 247, 258, 256, 262,
                    267, 270, 274, 278, 289, 290, 295, 298, 301, 305,
                    312, 318, 323, 330, 334, 338, 345, 345, 350, 356,
                    358, 362, 367], dtype=float)

    ax.scatter(sqft, kwh, s=40, color="#4C72B0", edgecolor="white", lw=0.8,
               zorder=5, label="Data", alpha=0.7)

    # Isotonic fit: simple pool-adjacent-violators (no sklearn needed)
    order = np.argsort(sqft)
    sqft_s, kwh_s = sqft[order], kwh[order].copy()
    # PAV algorithm
    n = len(kwh_s)
    iso = kwh_s.copy()
    i = 0
    while i < n - 1:
        if iso[i] > iso[i + 1]:
            # Pool
            j = i + 1
            while j < n and iso[j] <= iso[i]:
                j += 1
            pool_mean = np.mean(iso[i:j])
            iso[i:j] = pool_mean
            if i > 0:
                i -= 1
            continue
        i += 1
    ax.step(sqft_s, iso, where="post", color="#C44E52", lw=2.5,
            label="Isotonic fit")

    # OLS line for comparison
    m, b = np.polyfit(sqft, kwh, 1)
    ax.plot(sqft, m * sqft + b, color="#DD8452", ls="--", lw=1.5, label="OLS line")

    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Isotonic Regression: Non-Decreasing Fit")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "reg2_isotonic_fit.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ reg2_isotonic_fit.png")


# ── 24. tost_paired_diagram.png ───────────────────────────────────────────
def plot_tost_paired_diagram():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 3))

    # TOST paired results: mean_diff=0.27, CI=(-0.22, 0.76), delta=1.0
    delta = 1.0
    estimate = 0.27
    ci_lo = -0.22
    ci_hi = 0.76

    ax.axvline(-delta, color="#C44E52", ls="--", lw=2)
    ax.axvline(delta, color="#C44E52", ls="--", lw=2)
    ax.axvspan(-delta, delta, alpha=0.08, color="#55A868")

    ax.plot([ci_lo, ci_hi], [0.5, 0.5], color="#4C72B0", lw=3, solid_capstyle="round")
    ax.plot(estimate, 0.5, "o", color="#4C72B0", ms=10, zorder=5)

    ax.text(-delta, 1.1, f"−{delta}", ha="center", fontsize=10, color="#C44E52")
    ax.text(delta, 1.1, f"+{delta}", ha="center", fontsize=10, color="#C44E52")
    ax.text(estimate, 0.1, f"Δ = {estimate:.2f}", ha="center", fontsize=10, color="#4C72B0")
    ax.text(0, 1.5, "Equivalence Region", ha="center", fontsize=10,
            color="#55A868", fontweight="bold")
    ax.text(0, -0.7, "CI inside bounds → Equivalent (p < 0.05)",
            ha="center", fontsize=9, style="italic", color="#666")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 2)
    ax.set_xlabel("Paired Mean Difference")
    ax.set_title("TOST Paired Equivalence Test")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "tost_paired_diagram.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ tost_paired_diagram.png")


# ── 25. tost_comparison_table.png ─────────────────────────────────────────
def plot_tost_comparison_table():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    variants = [
        ["One-sample", "tost_t_test_one_sample", "0.0184", "Yes"],
        ["Two-sample", "tost_t_test_two_sample", "0.0017", "Yes"],
        ["Paired", "tost_t_test_paired", "0.0074", "Yes"],
        ["Correlation", "tost_correlation", "0.0002", "Yes"],
        ["Wilcoxon paired", "tost_wilcoxon_paired", "0.0156", "Yes"],
        ["Wilcoxon two-sample", "tost_wilcoxon_two_sample", "0.0547", "No"],
        ["Bootstrap", "tost_bootstrap", "< 0.05", "Yes"],
        ["Yuen (trimmed)", "tost_yuen", "0.0082", "Yes"],
    ]

    table = ax.table(
        cellText=variants,
        colLabels=["Variant", "Function", "TOST p-value", "Equivalent?"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color equivalence column
    for i in range(1, len(variants) + 1):
        if variants[i - 1][3] == "Yes":
            table[i, 3].set_facecolor("#E8F5E9")
        else:
            table[i, 3].set_facecolor("#FFEBEE")

    ax.set_title("TOST Equivalence Test Variants — Summary", fontsize=12,
                 fontweight="bold", pad=20)
    fig.savefig(OUT / "tost_comparison_table.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ tost_comparison_table.png")


# ── 26. cat_goodness_of_fit.png ───────────────────────────────────────────
def plot_cat_goodness_of_fit():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    faces = ["1", "2", "3", "4", "5", "6"]
    observed = [18, 22, 15, 25, 12, 28]
    expected = [20] * 6

    x = np.arange(len(faces))
    w = 0.35
    ax.bar(x - w / 2, observed, w, label="Observed", color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, expected, w, label="Expected", color="#DD8452",
           edgecolor="white", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.set_xlabel("Die Face")
    ax.set_ylabel("Count")
    ax.set_title("Chi-Square Goodness of Fit: Fair Die?")
    ax.legend(frameon=True, fancybox=True)
    ax.text(0.5, -0.12, "χ² = 8.60, p = 0.1262 — not enough evidence to reject fairness",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic", color="#666")
    fig.savefig(OUT / "cat_goodness_of_fit.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ cat_goodness_of_fit.png")


# ── 27. cat_agreement_heatmap.png ─────────────────────────────────────────
def plot_cat_agreement_heatmap():
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # 2x2 classification matrix: two raters
    matrix = np.array([[38, 5], [8, 49]])
    labels = ["Positive", "Negative"]

    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rater B")
    ax.set_ylabel("Rater A")
    ax.set_title("Inter-Rater Agreement")

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > 30 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.text(0.5, -0.18, "Cohen's κ = 0.7400 — substantial agreement",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic", color="#666")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "cat_agreement_heatmap.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ cat_agreement_heatmap.png")


# ── 28. fcast_loss_differential.png ───────────────────────────────────────
def plot_fcast_loss_differential():
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    np.random.seed(42)
    t = np.arange(200)
    # Simulated loss differential (model1 - model2)
    loss_diff = np.random.normal(0.3, 1.2, 200)
    # Add a regime shift at t=100
    loss_diff[100:] += 0.4

    ax.plot(t, loss_diff, color="#4C72B0", lw=0.8, alpha=0.7)
    ax.axhline(0, color="#C44E52", ls="--", lw=1.5)
    ax.axhline(np.mean(loss_diff), color="#55A868", ls="-", lw=2,
               label=f"Mean = {np.mean(loss_diff):.3f}")
    ax.fill_between(t, 0, loss_diff, where=loss_diff > 0,
                    alpha=0.15, color="#C44E52")
    ax.fill_between(t, 0, loss_diff, where=loss_diff < 0,
                    alpha=0.15, color="#4C72B0")

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Loss Differential (d_t)")
    ax.set_title("Diebold-Mariano: Loss Differential Series")
    ax.legend(frameon=True, fancybox=True)
    fig.savefig(OUT / "fcast_loss_differential.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ fcast_loss_differential.png")


# ── 29. fcast_mcs_bars.png ────────────────────────────────────────────────
def plot_fcast_mcs_bars():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    models = ["ARIMA", "ETS", "ML Model", "Random Walk"]
    included = [True, True, False, False]
    mse = [1.2, 1.35, 2.1, 2.8]
    colors = ["#55A868" if inc else "#C44E52" for inc in included]

    bars = ax.barh(models, mse, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("Mean Squared Error")
    ax.set_title("Model Confidence Set (α = 0.1)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#55A868", label="In MCS"),
                       Patch(facecolor="#C44E52", label="Excluded")]
    ax.legend(handles=legend_elements, frameon=True, fancybox=True, fontsize=9)

    for bar, v, inc in zip(bars, mse, included):
        label = f"{v:.2f}"
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=10)
    fig.savefig(OUT / "fcast_mcs_bars.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ fcast_mcs_bars.png")


# ── 30. cor2_nonlinear.png ────────────────────────────────────────────────
def plot_cor2_nonlinear():
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, 50)
    y = x**2 + rng.normal(0, 0.5, 50)

    ax.scatter(x, y, s=50, color="#4C72B0", edgecolor="white", lw=0.8, zorder=5)

    # Pearson line (linear fit)
    m, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(-3.5, 3.5, 100)
    ax.plot(x_fit, m * x_fit + b, color="#C44E52", ls="--", lw=1.5,
            label=f"Pearson r = −0.01 (linear)")
    # True relationship
    ax.plot(x_fit, x_fit**2, color="#55A868", lw=2,
            label=f"Distance cor = 0.76 (nonlinear)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Nonlinear Association: y = x² + noise")
    ax.legend(frameon=True, fancybox=True, fontsize=9)
    fig.savefig(OUT / "cor2_nonlinear.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ cor2_nonlinear.png")


# ── 31. cor2_icc_heatmap.png ──────────────────────────────────────────────
def plot_cor2_icc_heatmap():
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # 10 subjects × 3 raters
    ratings = np.array([
        [7, 8, 7],
        [6, 5, 6],
        [8, 9, 8],
        [5, 4, 5],
        [9, 9, 8],
        [7, 6, 7],
        [8, 8, 9],
        [6, 7, 6],
        [9, 8, 9],
        [7, 7, 7],
    ])

    im = ax.imshow(ratings, cmap="YlOrRd", aspect="auto", vmin=3, vmax=10)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Rater 1", "Rater 2", "Rater 3"])
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"S{i+1}" for i in range(10)])
    ax.set_ylabel("Subject")
    ax.set_title("Inter-Rater Reliability: 10 Subjects × 3 Raters")

    for i in range(10):
        for j in range(3):
            ax.text(j, i, str(ratings[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if ratings[i, j] >= 8 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Rating")
    fig.savefig(OUT / "cor2_icc_heatmap.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ cor2_icc_heatmap.png")


# ── 32. spec_alm_comparison.png ───────────────────────────────────────────
def plot_spec_alm_comparison():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y = np.array([10.2, 12.1, 11.5, 13.8, 50.0, 14.2, 11.8, 12.5, 15.1, 13.0,
                  16.5, 18.2, 17.0, 19.5, -20.0, 20.1, 17.8, 18.5, 21.2, 19.0,
                  22.5, 24.1, 23.0, 25.5, 80.0, 26.2, 23.8, 24.5, 27.1, 25.0,
                  28.5, 30.1, 29.0, 31.5, -10.0, 32.2, 29.8, 30.5, 33.1, 31.0])
    x1 = np.array([1.0, 1.5, 1.2, 2.0, 1.8, 2.2, 1.3, 1.7, 2.5, 1.9,
                   3.0, 3.5, 3.2, 4.0, 3.8, 4.2, 3.3, 3.7, 4.5, 3.9,
                   5.0, 5.5, 5.2, 6.0, 5.8, 6.2, 5.3, 5.7, 6.5, 5.9,
                   7.0, 7.5, 7.2, 8.0, 7.8, 8.2, 7.3, 7.7, 8.5, 7.9])

    # OLS: intercept=11.0407, coef_x1=2.3236 (distorted by outliers)
    ols_pred = 11.0407 + 2.3236 * x1
    # ALM (student_t): intercept=0.0, coef_x1=-4.6183, coef_x2=7.6061
    # Simplified: use x1 only for visualization
    x2 = np.array([2.0, 2.5, 2.2, 3.0, 2.8, 3.2, 2.3, 2.7, 3.5, 2.9,
                   4.0, 4.5, 4.2, 5.0, 4.8, 5.2, 4.3, 4.7, 5.5, 4.9,
                   6.0, 6.5, 6.2, 7.0, 6.8, 7.2, 6.3, 6.7, 7.5, 6.9,
                   8.0, 8.5, 8.2, 9.0, 8.8, 9.2, 8.3, 8.7, 9.5, 8.9])
    alm_pred = 0.0 + -4.6183 * x1 + 7.6061 * x2

    # Actual vs predicted
    ax = axes[0]
    outlier_mask = np.abs(y - np.median(y)) > 30
    ax.scatter(y[~outlier_mask], ols_pred[~outlier_mask], s=40, color="#C44E52",
               edgecolor="white", lw=0.6, alpha=0.7, label="OLS")
    ax.scatter(y[outlier_mask], ols_pred[outlier_mask], s=80, color="#C44E52",
               edgecolor="black", lw=1, marker="x", zorder=6)
    ax.scatter(y[~outlier_mask], alm_pred[~outlier_mask], s=40, color="#55A868",
               edgecolor="white", lw=0.6, alpha=0.7, label="ALM (Student-t)")
    ax.scatter(y[outlier_mask], alm_pred[outlier_mask], s=80, color="#55A868",
               edgecolor="black", lw=1, marker="x", zorder=6)
    lim = [-30, 85]
    ax.plot(lim, lim, ls="--", color="#999", lw=1)
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Actual vs Predicted")
    ax.legend(frameon=True, fancybox=True, fontsize=9)

    # Residuals comparison
    ax = axes[1]
    ols_resid = y - ols_pred
    alm_resid = y - alm_pred
    idx = np.arange(len(y))
    ax.scatter(idx, ols_resid, s=25, color="#C44E52", alpha=0.6,
               edgecolor="white", lw=0.4, label="OLS residuals")
    ax.scatter(idx, alm_resid, s=25, color="#55A868", alpha=0.6,
               edgecolor="white", lw=0.4, label="ALM residuals")
    ax.axhline(0, color="#999", ls="--", lw=1)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals: OLS vs ALM")
    ax.legend(frameon=True, fancybox=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "spec_alm_comparison.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ spec_alm_comparison.png")


# ── 33. spec_demand_timeline.png ──────────────────────────────────────────
def plot_spec_demand_timeline():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    demand = [0, 0, 5, 0, 0, 0, 8, 0, 0, 3,
              0, 0, 0, 12, 0, 0, 0, 0, 6, 0,
              0, 0, 0, 0, 15, 0, 0, 0, 0, 0,
              0, 4, 0, 0, 0, 0, 0, 9, 0, 0]

    t = np.arange(len(demand))
    colors = ["#4C72B0" if d > 0 else "#DDDDDD" for d in demand]
    # Mark high outlier (15.0)
    colors[24] = "#C44E52"

    ax.bar(t, demand, color=colors, edgecolor="white", linewidth=0.5)
    mean_val = np.mean(demand)
    ax.axhline(mean_val, color="#DD8452", ls="--", lw=1.5,
               label=f"Mean = {mean_val:.2f}")

    ax.set_xlabel("Time Period")
    ax.set_ylabel("Demand")
    ax.set_title("Intermittent Demand Pattern (80% zeros)")
    ax.legend(frameon=True, fancybox=True)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4C72B0", label="Normal demand"),
                       Patch(facecolor="#DDDDDD", label="Zero (stockout)"),
                       Patch(facecolor="#C44E52", label="High outlier")]
    ax.legend(handles=legend_elements, frameon=True, fancybox=True, fontsize=9,
              loc="upper left")
    fig.savefig(OUT / "spec_demand_timeline.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ spec_demand_timeline.png")


# ── 34. spec_dynamic_coefs.png ────────────────────────────────────────────
def plot_spec_dynamic_coefs():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    y = [10.5, 11.2, 10.8, 11.5, 12.0, 12.8, 13.5, 14.2, 15.0, 15.8,
         16.5, 17.2, 18.0, 18.8, 19.5, 20.2, 21.0, 21.8, 22.5, 23.2,
         28.0, 29.5, 31.0, 32.5, 34.0, 35.5, 37.0, 38.5, 40.0, 41.5,
         43.0, 44.5, 46.0, 47.5, 49.0, 50.5, 52.0, 53.5, 55.0, 56.5]
    x2 = [2.0, 2.2, 2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8,
          4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2,
          8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
          13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]

    t = np.arange(len(y))

    # Left: y over time with regime change
    ax = axes[0]
    ax.scatter(t[:20], y[:20], s=30, color="#4C72B0", edgecolor="white", label="Period 1")
    ax.scatter(t[20:], y[20:], s=30, color="#DD8452", edgecolor="white", label="Period 2")
    ax.axvline(19.5, color="#C44E52", ls="--", lw=1.5, alpha=0.7, label="Regime change")
    ax.set_xlabel("Time")
    ax.set_ylabel("y")
    ax.set_title("Response Over Time")
    ax.legend(frameon=True, fancybox=True, fontsize=9)

    # Right: x2 vs y with structural break
    ax = axes[1]
    ax.scatter(x2[:20], y[:20], s=30, color="#4C72B0", edgecolor="white", label="Period 1")
    ax.scatter(x2[20:], y[20:], s=30, color="#DD8452", edgecolor="white", label="Period 2")

    # Period 1 fit line
    m1, b1 = np.polyfit(x2[:20], y[:20], 1)
    x_fit1 = np.linspace(1.8, 6.5, 50)
    ax.plot(x_fit1, m1 * x_fit1 + b1, color="#4C72B0", lw=1.5, ls="--")
    # Period 2 fit line
    m2, b2 = np.polyfit(x2[20:], y[20:], 1)
    x_fit2 = np.linspace(7.5, 18, 50)
    ax.plot(x_fit2, m2 * x_fit2 + b2, color="#DD8452", lw=1.5, ls="--")

    ax.set_xlabel("x2")
    ax.set_ylabel("y")
    ax.set_title("Structural Break in x2 → y")
    ax.legend(frameon=True, fancybox=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "spec_dynamic_coefs.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓ spec_dynamic_coefs.png")


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating plots → {OUT}/")
    # Original 17 plots
    plot_hyp_distributions()
    plot_hyp_correlation()
    plot_reg_actual_vs_pred()
    plot_reg_residuals()
    plot_reg_quantile_lines()
    plot_grp_r2_bars()
    plot_grp_regression_lines()
    plot_abt_conversion_rates()
    plot_abt_tost_diagram()
    plot_hyp_boxplot_multigroup()
    plot_reg_coef_forest()
    plot_reg_ci_vs_pi()
    plot_reg_regularization_coefs()
    plot_grp_paired_before_after()
    plot_grp_aic_comparison()
    plot_abt_revenue_distributions()
    plot_abt_segment_forest()
    # New 17 plots
    plot_glm_poisson_fit()
    plot_glm_deviance_residuals()
    plot_glm_link_comparison()
    plot_reg2_lasso_coefs()
    plot_reg2_wls_residuals()
    plot_reg2_isotonic_fit()
    plot_tost_paired_diagram()
    plot_tost_comparison_table()
    plot_cat_goodness_of_fit()
    plot_cat_agreement_heatmap()
    plot_fcast_loss_differential()
    plot_fcast_mcs_bars()
    plot_cor2_nonlinear()
    plot_cor2_icc_heatmap()
    plot_spec_alm_comparison()
    plot_spec_demand_timeline()
    plot_spec_dynamic_coefs()
    print(f"Done — {len(list(OUT.glob('*.png')))} PNGs generated.")
