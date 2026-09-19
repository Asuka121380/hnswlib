"""Export actual V0 edge geometry and event estimates, then validate parity."""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

from common import mark_complete, read_json, sha256, write_json
from encode_sketch import open_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--exporter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    assets = contract["assets"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events_csv = output / "events.csv"
    edges_bin = output / "edges.bin"
    command = [args.exporter, "--index", assets["index"]["path"],
               "--sidecar", assets["sidecar"]["path"],
               "--queries", assets["queries"]["path"],
               "--input", assets["trace"]["path"],
               "--events", str(events_csv), "--edges", str(edges_bin),
               "--dimension", str(contract["config"]["dimension"])]
    subprocess.run(command, check=True)
    # The old trace was produced by a different numerical kernel. Its exact
    # distances still provide an independent check of IDs and vector layout.
    rows, max_exact_difference, max_current_difference = 0, 0.0, 0.0
    with events_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            rows += 1
            max_exact_difference = max(max_exact_difference,
                abs(float(row["D_exact_cpp"]) - float(row["old_exact"])))
            max_current_difference = max(max_current_difference,
                abs(float(row["Dc_exact_cpp"]) - float(row["old_current"])))
    if not rows or max_exact_difference > 1e-4 or max_current_difference > 1e-4:
        raise ValueError("historical trace geometry did not close; inspect IDs and index")
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("PyArrow is required for the canonical events.parquet") from error
    table = pacsv.read_csv(events_csv)
    splits = read_json(contract["split_path"])
    split_for_query = {query_id: name for name, ids in splits.items()
                       for query_id in ids}
    query_ids = table["query_id"].to_numpy()
    tau = table["tau"].to_numpy()
    exact = table["D_exact_cpp"].to_numpy()
    if not np.all(np.isfinite(tau) & (tau > 0) & np.isfinite(exact)):
        raise ValueError("invalid event threshold or exact distance")
    event_splits = [split_for_query[int(query_id)] for query_id in query_ids]
    eligible = table["edge_eligible"].to_numpy().astype(bool)
    table = table.append_column("event_id", pa.array(np.arange(rows, dtype=np.uint64)))
    table = table.append_column("source_run", pa.array(["historical_phase1"] * rows))
    table = table.append_column("split", pa.array(event_splits))
    table = table.append_column("graph_layer", pa.array(np.zeros(rows, dtype=np.uint32)))
    table = table.append_column("ef_search", pa.nulls(rows, type=pa.uint32()))
    table = table.append_column("threshold_valid", pa.array(np.ones(rows, dtype=bool)))
    table = table.append_column("Dhat_pq_ref", pa.nulls(rows, type=pa.float64()))
    table = table.append_column("invalid_reason", pa.array(
        [None if item else "raw_invalid" for item in eligible]))
    table = table.append_column("oracle_far", pa.array(exact > tau))
    table = table.append_column("sampling_modulus", pa.array(
        np.full(rows, 16, dtype=np.uint32)))
    table = table.append_column("sampling_remainder", pa.nulls(rows, type=pa.uint32()))
    table = table.append_column("event_sequence_if_available", pa.nulls(
        rows, type=pa.uint64()))
    table = table.append_column("stage_fields_available", pa.array(
        np.zeros(rows, dtype=bool)))
    pq.write_table(table, output / "events.parquet", compression="zstd")
    edge_data = open_edges(edges_bin, contract["config"]["dimension"])
    edge_schema = pa.schema([
        ("edge_index", pa.uint64()), ("edge_ordinal", pa.uint64()),
        ("current_internal_id", pa.uint32()),
        ("candidate_internal_id", pa.uint32()),
        ("neighbor_slot", pa.uint32()), ("flags", pa.uint32()),
        ("code", pa.binary()), ("ell_s", pa.float64()),
        ("a_s", pa.float64()), ("z_norm", pa.float64()),
        ("B", pa.float64()),
    ])
    with pq.ParquetWriter(output / "edges.parquet", edge_schema,
                          compression="zstd") as writer:
        for start in range(0, len(edge_data), 4096):
            batch = edge_data[start:start + 4096]
            columns = [
                np.arange(start, start + len(batch), dtype=np.uint64),
                batch["ordinal"], batch["source"], batch["target"],
                batch["slot"], batch["flags"],
                [bytes(code) for code in batch["code"]],
                batch["length"], batch["anchor"],
                np.linalg.norm(batch["residual"], axis=1), batch["bias"],
            ]
            writer.write_table(pa.Table.from_arrays(
                [pa.array(column, type=field.type)
                 for column, field in zip(columns, edge_schema)],
                schema=edge_schema))
    manifest = {"schema_version": 1, "contract": str(Path(args.contract).resolve()),
                "events_csv": str(events_csv.resolve()),
                "events_parquet": str((output / "events.parquet").resolve()),
                "edges_parquet": str((output / "edges.parquet").resolve()),
                "edges_bin": str(edges_bin.resolve()), "event_count": rows,
                "unique_edge_count": len(edge_data),
                "max_exact_difference": max_exact_difference,
                "max_current_difference": max_current_difference,
                "events_sha256": sha256(events_csv),
                "edges_sha256": sha256(edges_bin)}
    manifest["edges_parquet_sha256"] = sha256(output / "edges.parquet")
    write_json(output / "manifest.json", manifest)
    mark_complete(output, {"contract": sha256(args.contract)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
