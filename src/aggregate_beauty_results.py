"""Aggregate multi-seed Beauty results and paired AdaFuse comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "beauty"
DEFAULT_SEEDS = (42, 123, 456, 789, 1024)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def load_runs(results_dir, seeds):
    runs = {}
    for seed in seeds:
        path = results_dir / f"seed_{seed}" / "results.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed result: {path}")
        with path.open("rb") as handle:
            runs[seed] = pickle.load(handle)
    return runs


def aggregate(runs):
    seeds = list(runs)
    common_models = set.intersection(
        *(set(runs[seed]["aggregate"]) for seed in seeds)
    )
    metrics = ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "HR@10")
    summary = {}
    for model in sorted(common_models):
        summary[model] = {}
        for metric in metrics:
            values = np.array(
                [runs[seed]["aggregate"][model][metric] for seed in seeds], dtype=float
            )
            summary[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def paired_tests(runs):
    """Average each user over seeds, then run paired tests across users."""
    comparisons = {}
    models = set.intersection(
        *(set(payload["per_user"]) for payload in runs.values())
    )
    if "adafuse" not in models:
        return comparisons
    for model in sorted(models - {"adafuse"}):
        comparisons[model] = {}
        for metric in ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20"):
            common_users = set.intersection(
                *(
                    set(payload["per_user"]["adafuse"])
                    & set(payload["per_user"][model])
                    for payload in runs.values()
                )
            )
            adafuse_values = [
                np.mean(
                    [payload["per_user"]["adafuse"][user][metric] for payload in runs.values()]
                )
                for user in sorted(common_users)
            ]
            baseline_values = [
                np.mean(
                    [payload["per_user"][model][user][metric] for payload in runs.values()]
                )
                for user in sorted(common_users)
            ]
            ada = np.asarray(adafuse_values, dtype=float)
            base = np.asarray(baseline_values, dtype=float)
            difference = ada - base
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = stats.ttest_rel(ada, base)
                statistic, p_value = float(test.statistic), float(test.pvalue)
            comparisons[model][metric] = {
                "mean_difference": float(difference.mean()),
                "t_statistic": statistic,
                "p_value": p_value,
                "significant_at_0.05": bool(p_value < 0.05),
                "observations": int(len(difference)),
            }
    return comparisons


def alpha_analysis(runs):
    user_values = {}
    for payload in runs.values():
        for user, record in payload.get("alphas", {}).items():
            user = int(user)
            entry = user_values.setdefault(
                user, {"alphas": [], "activity": int(record["train_interactions"])}
            )
            if entry["activity"] != int(record["train_interactions"]):
                raise ValueError("activity count changed across seeds")
            entry["alphas"].append(float(record["alpha"]))
    if not user_values:
        return {"error": "no alpha values were saved"}
    rows = [
        (user, float(np.mean(record["alphas"])), record["activity"])
        for user, record in sorted(user_values.items())
    ]
    alpha = np.asarray([row[1] for row in rows])
    activity = np.asarray([row[2] for row in rows])
    pearson = stats.pearsonr(activity, alpha)
    spearman = stats.spearmanr(activity, alpha)
    groups = {"1-5": [], "6-10": [], "11-20": [], "21-50": [], "50+": []}
    for _, value, count in rows:
        if count <= 5:
            group = "1-5"
        elif count <= 10:
            group = "6-10"
        elif count <= 20:
            group = "11-20"
        elif count <= 50:
            group = "21-50"
        else:
            group = "50+"
        groups[group].append(value)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "groups": {
            group: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for group, values in groups.items()
        },
    }


def write_outputs(results_dir, seeds, summary, tests, alpha):
    tables = results_dir / "comparison"
    tables.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": seeds,
        "aggregate": summary,
        "paired_tests_adafuse_vs_baseline": tests,
        "alpha_analysis": alpha,
    }
    with (tables / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (tables / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "metric", "mean", "std", "formatted"))
        for model, metrics in summary.items():
            for metric, values in metrics.items():
                writer.writerow(
                    (
                        model,
                        metric,
                        values["mean"],
                        values["std"],
                        f"{values['mean']:.4f} +/- {values['std']:.4f}",
                    )
                )
    return tables


def main():
    args = parse_args()
    runs = load_runs(args.results_dir, args.seeds)
    protocols = []
    for run in runs.values():
        protocol = dict(run["protocol"])
        protocol.pop("seed", None)
        arguments = dict(protocol.get("arguments", {}))
        arguments.pop("seed", None)
        protocol["arguments"] = arguments
        protocols.append(json.dumps(protocol, sort_keys=True))
    if len(set(protocols)) != 1:
        raise ValueError("seed runs do not share an identical protocol")
    summary = aggregate(runs)
    tests = paired_tests(runs)
    alpha = alpha_analysis(runs)
    output = write_outputs(args.results_dir, args.seeds, summary, tests, alpha)
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
