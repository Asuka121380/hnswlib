"""Produce exportable recall/QPS and exact-distance diagnostic figures."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import cli,unseal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--analysis",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();data=unseal(a.analysis);dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method in ("hnsw","opq","ivf_opq"):
        values=sorted([r for r in data["points"] if r["case"]["method"]==method],key=lambda r:r["recall_at_k"])
        ax.plot([r["recall_at_k"] for r in values],[r["qps_service"] for r in values],"o-",label=method)
    ax.set(xlabel="Recall@10",ylabel="QPS (full service wall)",title=data["dataset_id"]+(" — LOCAL SMOKE" if data["local_smoke"] else ""))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for suffix in ("png","pdf"):fig.savefig(dest/("recall_qps."+suffix),dpi=200)
    plt.close(fig)
    if data.get("diagnostics"):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in ("hnsw","opq","ivf_opq"):
            values=sorted([r for r in data["diagnostics"] if r["case"]["method"]==method],key=lambda r:r["recall_at_k"])
            ax.plot([r["recall_at_k"] for r in values],[r["exact_l2_calls_per_query"] for r in values],"o-",label=method)
        ax.set(xlabel="Recall@10",ylabel="Actual exact L2 calls per query")
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ("png","pdf"):fig.savefig(dest/("exact_distances."+suffix),dpi=200)
        plt.close(fig)
if __name__=="__main__":cli(main)
