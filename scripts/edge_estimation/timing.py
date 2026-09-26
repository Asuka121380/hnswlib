from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def paired_schedule(methods: Sequence[str], blocks: int = 5,
                    repeats: int = 5, seed: int = 20260924) -> list[dict[str, object]]:
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("methods must be non-empty and unique")
    if blocks <= 0 or repeats <= 0:
        raise ValueError("blocks and repeats must be positive")
    rng = random.Random(seed)
    schedule: list[dict[str, object]] = []
    for block in range(blocks):
        order = list(methods)
        rng.shuffle(order)
        for repeat in range(repeats):
            for position, method in enumerate(order):
                schedule.append({"block": block, "repeat": repeat,
                                 "position": position, "method": method})
    return schedule


def summarize_paired(records: Sequence[Mapping[str, object]],
                     reference: str) -> Mapping[str, object]:
    if not records:
        raise ValueError("timing records are empty")
    identities = {(record["dataset_id"], record["build_id"], record["mode"],
                   record["thread_count"], record["isa_profile"])
                  for record in records}
    if len(identities) != 1:
        raise ValueError("paired timing records do not share one execution contract")
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    query_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    for record in records:
        elapsed = float(record["elapsed_ns"])
        queries = int(record["query_count"])
        events = int(record["eligible_events"])
        method = str(record["method"])
        if elapsed < 0 or queries <= 0 or events < 0:
            raise ValueError("invalid raw timing record")
        grouped[(int(record["block"]), method)].append(elapsed)
        query_counts.setdefault(method, queries)
        event_counts.setdefault(method, events)
        if query_counts[method] != queries or event_counts[method] != events:
            raise ValueError("method work counts changed across paired records")
    methods = sorted({method for _, method in grouped})
    if reference not in methods:
        raise ValueError("reference method is absent")
    block_medians = {key: statistics.median(values) for key, values in grouped.items()}
    blocks = sorted({block for block, _ in grouped})
    for block in blocks:
        if any((block, method) not in block_medians for method in methods):
            raise ValueError("paired block is incomplete")
    summaries: dict[str, object] = {}
    for method in methods:
        medians = [block_medians[(block, method)] for block in blocks]
        ratios = [block_medians[(block, reference)] / block_medians[(block, method)]
                  for block in blocks]
        summaries[method] = {
            "block_median_elapsed_ns": medians,
            "median_elapsed_ns": statistics.median(medians),
            "ns_per_query": statistics.median(medians) / query_counts[method],
            "ns_per_eligible_edge": (
                None if event_counts[method] == 0
                else statistics.median(medians) / event_counts[method]
            ),
            "paired_speed_ratio_vs_reference": statistics.median(ratios),
        }
    return {"schema_version": 1, "reference": reference,
            "block_count": len(blocks), "methods": summaries}
