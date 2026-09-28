"""OPQ confirmation study: paired beta coverage, repeats, then fixed-parameter batching."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.edge_estimation.ivf_experiments import expanded_cases

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (.95, .97, .98)


def make_config(base, cases, purpose):
    cfg = {**base, "purpose": purpose, "repeats": 3, "cases": cases}
    cfg.pop("batch_grid", None)
    cfg["methods"] = {}
    for method in sorted({r["method"] for r in cases}):
        values = [r for r in cases if r["method"] == method]
        cfg["methods"][method] = {"ef_search": sorted({r["ef_search"] for r in values})}
        if method != "hnsw":
            cfg["methods"][method]["beta"] = sorted({r["beta"] for r in values})
    expanded_cases(cfg)
    return cfg


def confirmation_config(base):
    cases = []
    for ef in (350, 375, 400, 550, 575, 600, 850, 900, 950):
        cases.append(dict(method="hnsw", ef_search=ef, beta=None, batch_size=1))
    for method in ("opq", "ivf_opq"):
        for beta in (1.2, 1.3, 1.35):
            efs = ((425, 450, 475, 650, 675, 700, 1000, 1075, 1150) if beta == 1.2
                   else (350, 375, 400, 575, 600, 625, 900, 950, 1000))
            for ef in efs:
                cases.append(dict(method=method, ef_search=ef, beta=beta, batch_size=128))
    return make_config(base, cases, "63 configurations x 3 repeats; symmetric old/new OPQ beta coverage")


def select_batch_config(base, rows):
    cases = {}
    selections = []
    for target in TARGETS:
        for method in ("hnsw", "opq", "ivf_opq"):
            candidates = [r for r in rows if r["method"] == method and float(r["recall_at_k"]) >= target-1e-12]
            if not candidates:
                selections.append(dict(target=target, method=method, status="NOT_REACHED"))
                continue
            close = [r for r in candidates if float(r["recall_at_k"]) <= target+.002+1e-12]
            if close:
                row = max(close, key=lambda r: float(r["qps"]))
                status = "WITHIN_0.2_PERCENTAGE_POINTS"
            else:
                row = min(candidates, key=lambda r: (float(r["recall_at_k"]), -float(r["qps"])))
                status = "OUTSIDE_RECALL_WINDOW"
            beta = None if method == "hnsw" else float(row["beta"])
            ef = int(row["ef_search"])
            selections.append(dict(target=target, method=method, status=status, beta=beta,
                                   ef_search=ef, recall=float(row["recall_at_k"]), qps_median=float(row["qps"])))
            for batch in ([1] if method == "hnsw" else [1, 32, 128, 256]):
                cases[(method, ef, beta, batch)] = dict(method=method, ef_search=ef, beta=beta, batch_size=batch)
    return make_config(base, list(cases.values()), "Fixed beta/ef across batch sizes; 3 repeats"), selections


def main():
    run = Path(sys.argv[1]).resolve()
    common = sys.argv[2:]
    base = json.loads((ROOT/"configs/edge_estimation/ivf_k256_controlled_full.json").read_text())
    configs = run/"opq-configs"
    configs.mkdir()
    driver = ROOT/"scripts/edge_estimation/ivf_experiments.py"

    def execute(name, cfg):
        (run/"OPQ_STAGE.txt").write_text(name+"\n")
        path = configs/f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2)+"\n")
        count = len(expanded_cases(cfg))
        print(f"OPQ_STAGE={name} runs={count}", flush=True)
        subprocess.run([sys.executable, str(driver), "grid", "--config", str(path),
                        *common, "--out", str(run/name)], check=True)
        subprocess.run([sys.executable, str(driver), "summarize",
                        "--out", str(run/name), "--plot"], check=True)
        actual = json.loads((run/name/"complete.json").read_text())["points"]
        assert actual == count
        return count

    confirmation = confirmation_config(base)
    n_confirm = execute("opq-confirm", confirmation)
    with (run/"opq-confirm/aggregated.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    batch_cfg, selected = select_batch_config(base, rows)
    (run/"opq-selected.json").write_text(json.dumps(selected, indent=2)+"\n")
    n_batch = execute("opq-batch", batch_cfg)

    with (run/"opq-batch/aggregated.csv").open() as stream:
        batch_rows = list(csv.DictReader(stream))
    lines = ["# OPQ confirmation and query-batch study", "",
             f"Confirmation: {n_confirm} runs (63 configurations, 3 repeats).",
             f"Batch: {n_batch} runs; parameters fixed across batch=1/32/128/256.", "",
             "Selection window: target to target+0.2 percentage points. Outside-window points are explicitly marked.",
             "Exploratory queries reused. Selection and repeated timings are not an independent held-out evaluation.",
             "Both old and IVF OPQ include beta=1.20; static false-pruning gates are not asserted here.", "",
             "| Target | Method | Selection | beta | ef | Recall | Median QPS (batch128 for OPQ) |",
             "|---|---|---|---|---|---|---|"]
    for row in selected:
        lines.append("| "+" | ".join(str(row.get(k,"")) for k in
                     ("target","method","status","beta","ef_search","recall","qps_median"))+" |")
    lines += ["", "## Fixed-parameter batch results", "",
              "| Method | beta | ef | batch | Recall | Median QPS | Std QPS |",
              "|---|---|---|---|---|---|---|"]
    for row in sorted(batch_rows, key=lambda r:(r["method"],float(r["beta"] or 0),int(r["ef_search"]),int(r["batch_size"]))):
        lines.append("| "+" | ".join(str(row[k]) for k in
                     ("method","beta","ef_search","batch_size","recall_at_k","qps","qps_std"))+" |")
    (run/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    (run/"COMPLETE.json").write_text(json.dumps(dict(valid=True, study="opq-confirm",
        confirmation_runs=n_confirm, batch_runs=n_batch, smoke_points=7,
        source_commit=(run/"source-commit.txt").read_text().strip()), indent=2)+"\n")
    (run/"OPQ_STAGE.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
