"""
Paired t-tests comparing recall values across methods.

Structure: 3 methods x 4 datasets x 2 sorters x 5 splits = 120 observations.
Tests run on 8 dataset×sorter means per method.

Comparisons (H1: A > B):
  1. slay_auto > si
  2. slay_auto > si_auto
  3. si_auto > si

Holm correction applied across 3 comparisons.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULTS_DIR = Path(__file__).parent.parent / "results" / "artificial_splits"
METHODS = ("si", "si_auto", "slay_auto")
COMPARISONS = [
    ("slay_auto", "si"),
    ("slay_auto", "si_auto"),
    ("si_auto", "si"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def parse_filename(stem: str) -> tuple[str, str]:
    for sorter in ("ks25", "ks4"):
        if stem.endswith(f"_{sorter}"):
            return stem[: -(len(sorter) + 1)], sorter
    raise ValueError(f"Cannot parse sorter from filename: {stem}")


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        dataset, sorter = parse_filename(path.stem)
        data = json.loads(path.read_text())
        for method in METHODS:
            for split_id, recall in enumerate(data[method]["raw_recall"]):
                rows.append(
                    dict(
                        recall=recall,
                        method=method,
                        dataset=dataset,
                        sorter=sorter,
                        split_id=split_id,
                    )
                )
    df = pd.DataFrame(rows)
    assert len(df) == 120, f"Expected 120 rows, got {len(df)}"
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "     n/a"
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def direction_counts(wide: pd.DataFrame, a: str, b: str) -> tuple[int, int, int]:
    diff = wide[a] - wide[b]
    return (diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum()


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = stats.ttest_rel(a, b, alternative="greater")
    return res.statistic, res.pvalue


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a - b
    return diffs.mean() / diffs.std(ddof=1)


def cohens_d_ci(d: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for paired Cohen's d via noncentral t noncentrality parameter inversion."""
    from scipy.optimize import brentq

    t_obs = d * np.sqrt(n)
    df = n - 1
    # Find nc_lower s.t. P(T(df, nc) >= t_obs) = 1 - alpha/2  →  d_lower
    # Find nc_upper s.t. P(T(df, nc) >= t_obs) = alpha/2      →  d_upper
    nc_lower = brentq(lambda nc: stats.nct.sf(t_obs, df, nc) - (1 - alpha / 2), -50, 50)
    nc_upper = brentq(lambda nc: stats.nct.sf(t_obs, df, nc) - alpha / 2, -50, 50)
    return nc_lower / np.sqrt(n), nc_upper / np.sqrt(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_dataframe()
    print(
        f"Loaded {len(df)} rows across "
        f"{df['dataset'].nunique()} datasets, "
        f"{df['sorter'].nunique()} sorters, "
        f"{df['method'].nunique()} methods, "
        f"{df['split_id'].nunique()} splits.\n"
    )

    # Dataset×sorter means (8 rows per method)
    means = df.groupby(["dataset", "sorter", "method"])["recall"].mean().reset_index()
    wide_means = means.pivot_table(
        index=["dataset", "sorter"], columns="method", values="recall"
    ).reset_index()

    results = []

    for method_a, method_b in COMPARISONS:
        label = f"{method_a} > {method_b}"
        print(f"{'=' * 65}")
        print(f"  {label}")
        print(f"{'=' * 65}")

        n_gt_ds, n_eq_ds, n_lt_ds = direction_counts(wide_means, method_a, method_b)
        print(f"\n  Dataset×sorter (n=8): A>B={n_gt_ds}  A==B={n_eq_ds}  A<B={n_lt_ds}")

        diffs_8 = (wide_means[method_a] - wide_means[method_b]).values
        labels_8 = (wide_means["dataset"] + "_" + wide_means["sorter"]).values
        sorted_idx = np.argsort(diffs_8)
        print("\n  Dataset×sorter mean differences (sorted):")
        for i in sorted_idx:
            print(f"    {labels_8[i]:<35} {diffs_8[i]:+.4f}")
        mean_diff = diffs_8.mean()
        print(f"  Mean of 8 differences: {mean_diff:+.4f}")

        a8 = wide_means[method_a].values
        b8 = wide_means[method_b].values
        t_stat, p_t = paired_ttest(a8, b8)
        d = cohens_d_paired(a8, b8)
        d_lo, d_hi = cohens_d_ci(d, n=len(a8))
        print(
            f"\n  Paired t-test: t={t_stat:.4f}  p={fmt_p(p_t)}  d={d:.3f} [{d_lo:.3f}, {d_hi:.3f}]"
        )

        results.append(
            dict(
                label=label,
                n_gt_ds=n_gt_ds,
                n_eq_ds=n_eq_ds,
                mean_diff=mean_diff,
                t_stat=t_stat,
                p_t=p_t,
                cohens_d=d,
                d_lo=d_lo,
                d_hi=d_hi,
            )
        )

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    _, p_t_holm, _, _ = multipletests([r["p_t"] for r in results], method="holm")

    print(f"\n{'=' * 65}")
    print("SUMMARY TABLE (Holm-corrected p shown in brackets)\n")

    col_w = 22
    h = (
        f"{'Comparison':<{col_w}} {'Datasets':>9} "
        f"{'MeanDiff':>9} {'t':>7} {'t-test p':>20} {'d [95% CI]'}"
    )
    print(h)
    print("-" * len(h))

    for i, r in enumerate(results):
        datasets_str = f"{r['n_gt_ds']}/8" + (
            f"({r['n_eq_ds']}tie)" if r["n_eq_ds"] else ""
        )

        def fmt_pair(p_raw, p_corr):
            sig = "*" if p_corr < 0.05 else ""
            return f"{fmt_p(p_raw)} [{fmt_p(p_corr)}]{sig}"

        print(
            f"{r['label']:<{col_w}} {datasets_str:>9} "
            f"{r['mean_diff']:>+9.4f} "
            f"{r['t_stat']:>7.3f} "
            f"{fmt_pair(r['p_t'], p_t_holm[i]):>20} "
            f"{r['cohens_d']:.3f} [{r['d_lo']:.3f}, {r['d_hi']:.3f}]"
        )
