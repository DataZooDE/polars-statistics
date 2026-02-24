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


# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating plots → {OUT}/")
    plot_hyp_distributions()
    plot_hyp_correlation()
    plot_reg_actual_vs_pred()
    plot_reg_residuals()
    plot_reg_quantile_lines()
    plot_grp_r2_bars()
    plot_grp_regression_lines()
    plot_abt_conversion_rates()
    plot_abt_tost_diagram()
    print(f"Done — {len(list(OUT.glob('*.png')))} PNGs generated.")
