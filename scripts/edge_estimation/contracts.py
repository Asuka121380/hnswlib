from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import struct
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1
HEADER_SIZE = 64
EVENT_RECORD_SIZE = 80
LABEL_RECORD_SIZE = 16
QUERY_RANGE_RECORD_SIZE = 24
INVALID_U64 = (1 << 64) - 1
INVALID_U32 = (1 << 32) - 1

HEADER = struct.Struct("<8sIIIIQ32s")
EVENT = struct.Struct("<BBHiQQQQIIIIddQ")
LABEL = struct.Struct("<Qd")
QUERY_RANGE = struct.Struct("<QQQ")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class HeaderRecord:
    magic: bytes
    schema_version: int
    header_size: int
    record_size: int
    dimension: int
    record_count: int
    identity: bytes


@dataclass(frozen=True)
class EventRecord:
    kind: int
    flags: int
    graph_layer: int
    event_id: int
    query_id: int
    expansion_id: int = INVALID_U64
    edge_id: int = INVALID_U64
    source_id: int = INVALID_U32
    target_id: int = INVALID_U32
    neighbor_slot: int = INVALID_U32
    source_degree: int = 0
    d_current: float = 0.0
    threshold_before: float = 0.0


@dataclass(frozen=True)
class LabelRecord:
    event_id: int
    exact_squared_distance: float


@dataclass(frozen=True)
class QueryRangeRecord:
    query_id: int
    begin_event: int
    event_count: int


def _no_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_no_duplicate_pairs)


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def atomic_output_dir(destination: os.PathLike[str] | str) -> Iterator[Path]:
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"output already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(
        prefix=f".{destination_path.name}.partial-", dir=destination_path.parent
    ))
    try:
        yield partial
        os.replace(partial, destination_path)
    except BaseException:
        # Preserve the partial directory as forensic evidence.
        raise


def _header_bytes(magic: bytes, record_size: int, dimension: int,
                  count: int, identity: bytes) -> bytes:
    if len(magic) != 8 or len(identity) != 32:
        raise ContractError("magic and identity must be 8 and 32 bytes")
    return HEADER.pack(magic, SCHEMA_VERSION, HEADER_SIZE, record_size,
                       dimension, count, identity)


def _read_records(path: os.PathLike[str] | str, magic: bytes,
                  record_size: int) -> tuple[HeaderRecord, bytes]:
    data = Path(path).read_bytes()
    if len(data) < HEADER_SIZE:
        raise ContractError("truncated binary header")
    values = HEADER.unpack_from(data)
    header = HeaderRecord(*values)
    if (header.magic != magic or header.schema_version != SCHEMA_VERSION or
            header.header_size != HEADER_SIZE or header.record_size != record_size):
        raise ContractError("unsupported binary header")
    expected = HEADER_SIZE + header.record_count * record_size
    if expected != len(data):
        raise ContractError("binary size mismatch")
    return header, data[HEADER_SIZE:]


def validate_events(events: Sequence[EventRecord]) -> None:
    in_query = False
    in_source = False
    query_id = None
    expansion_id = None
    for index, event in enumerate(events):
        if event.event_id != index:
            raise ContractError("event IDs must be contiguous")
        if event.flags & ~0x07:
            raise ContractError("reserved event flag set")
        if event.kind == 1:
            if (in_query or event.graph_layer != -1 or event.flags or
                    event.expansion_id != INVALID_U64 or event.edge_id != INVALID_U64 or
                    event.source_id != INVALID_U32 or event.target_id != INVALID_U32 or
                    event.neighbor_slot != INVALID_U32 or event.source_degree or
                    event.d_current != 0.0 or event.threshold_before != 0.0):
                raise ContractError("invalid QUERY_BEGIN")
            in_query, in_source, query_id = True, False, event.query_id
        elif event.kind == 2:
            if (not in_query or event.query_id != query_id or event.graph_layer != -1 or
                    event.flags or event.expansion_id == INVALID_U64 or
                    event.edge_id != INVALID_U64 or event.source_id == INVALID_U32 or
                    event.target_id != INVALID_U32 or event.neighbor_slot != INVALID_U32 or
                    not math.isfinite(event.d_current) or event.d_current < 0 or
                    event.threshold_before != 0.0):
                raise ContractError("invalid SOURCE_BEGIN")
            in_source, expansion_id = True, event.expansion_id
        elif event.kind == 3:
            if (not in_query or not in_source or event.query_id != query_id or
                    event.expansion_id != expansion_id or event.graph_layer != 0 or
                    event.edge_id == INVALID_U64 or event.source_id == INVALID_U32 or
                    event.target_id == INVALID_U32 or event.neighbor_slot == INVALID_U32 or
                    event.neighbor_slot >= event.source_degree or not (event.flags & 2) or
                    ((event.flags & 4) and not (event.flags & 1)) or
                    not math.isfinite(event.d_current) or event.d_current < 0):
                raise ContractError("invalid CANDIDATE")
            if not (event.flags & 1) and event.threshold_before != 0.0:
                raise ContractError("invalid threshold encoding")
            if ((event.flags & 1) and
                    (not math.isfinite(event.threshold_before) or event.threshold_before < 0)):
                raise ContractError("invalid threshold value")
        elif event.kind == 4:
            if (not in_query or event.query_id != query_id or event.flags or
                    event.edge_id != INVALID_U64 or event.target_id == INVALID_U32 or
                    event.neighbor_slot != INVALID_U32 or event.threshold_before != 0.0):
                raise ContractError("invalid EXACT_ONLY")
        elif event.kind == 5:
            if (not in_query or event.query_id != query_id or event.graph_layer != -1 or
                    event.flags or event.expansion_id != INVALID_U64 or
                    event.edge_id != INVALID_U64 or event.source_id != INVALID_U32 or
                    event.target_id != INVALID_U32 or event.neighbor_slot != INVALID_U32 or
                    event.source_degree or event.d_current != 0.0 or
                    event.threshold_before != 0.0):
                raise ContractError("invalid QUERY_END")
            in_query, in_source = False, False
        else:
            raise ContractError(f"unknown event kind: {event.kind}")
    if in_query:
        raise ContractError("unterminated query")


def write_events(path: os.PathLike[str] | str, dimension: int,
                 identity: bytes, events: Sequence[EventRecord]) -> None:
    validate_events(events)
    with Path(path).open("wb") as stream:
        stream.write(_header_bytes(b"UQEV0001", EVENT_RECORD_SIZE,
                                   dimension, len(events), identity))
        for item in events:
            stream.write(EVENT.pack(
                item.kind, item.flags, 0, item.graph_layer, item.event_id,
                item.query_id, item.expansion_id, item.edge_id,
                item.source_id, item.target_id, item.neighbor_slot,
                item.source_degree, item.d_current, item.threshold_before, 0
            ))


def read_events(path: os.PathLike[str] | str) -> tuple[HeaderRecord, list[EventRecord]]:
    header, payload = _read_records(path, b"UQEV0001", EVENT_RECORD_SIZE)
    result: list[EventRecord] = []
    for values in EVENT.iter_unpack(payload):
        (kind, flags, reserved16, graph_layer, event_id, query_id,
         expansion_id, edge_id, source_id, target_id, neighbor_slot,
         source_degree, d_current, threshold_before, reserved64) = values
        if reserved16 or reserved64:
            raise ContractError("reserved event field is non-zero")
        result.append(EventRecord(
            kind, flags, graph_layer, event_id, query_id, expansion_id,
            edge_id, source_id, target_id, neighbor_slot, source_degree,
            d_current, threshold_before
        ))
    validate_events(result)
    return header, result


def write_labels(path: os.PathLike[str] | str, dimension: int,
                 identity: bytes, labels: Sequence[LabelRecord]) -> None:
    previous = -1
    with Path(path).open("wb") as stream:
        stream.write(_header_bytes(b"UQLB0001", LABEL_RECORD_SIZE,
                                   dimension, len(labels), identity))
        for item in labels:
            if item.event_id <= previous or not math.isfinite(item.exact_squared_distance):
                raise ContractError("invalid label record")
            previous = item.event_id
            stream.write(LABEL.pack(item.event_id, item.exact_squared_distance))


def read_labels(path: os.PathLike[str] | str) -> tuple[HeaderRecord, list[LabelRecord]]:
    header, payload = _read_records(path, b"UQLB0001", LABEL_RECORD_SIZE)
    labels = [LabelRecord(*values) for values in LABEL.iter_unpack(payload)]
    previous = -1
    for item in labels:
        if item.event_id <= previous or not math.isfinite(item.exact_squared_distance):
            raise ContractError("invalid label record")
        previous = item.event_id
    return header, labels


def write_query_ranges(path: os.PathLike[str] | str, dimension: int,
                       identity: bytes,
                       ranges: Sequence[QueryRangeRecord]) -> None:
    expected = 0
    with Path(path).open("wb") as stream:
        stream.write(_header_bytes(b"UQQR0001", QUERY_RANGE_RECORD_SIZE,
                                   dimension, len(ranges), identity))
        for item in ranges:
            if item.begin_event != expected:
                raise ContractError("query ranges must be contiguous")
            expected += item.event_count
            stream.write(QUERY_RANGE.pack(item.query_id, item.begin_event,
                                          item.event_count))


def read_query_ranges(path: os.PathLike[str] | str) -> tuple[HeaderRecord, list[QueryRangeRecord]]:
    header, payload = _read_records(path, b"UQQR0001", QUERY_RANGE_RECORD_SIZE)
    ranges = [QueryRangeRecord(*values) for values in QUERY_RANGE.iter_unpack(payload)]
    expected = 0
    for item in ranges:
        if item.begin_event != expected:
            raise ContractError("query ranges must be contiguous")
        expected += item.event_count
    return header, ranges


def validate_dataset(events: Sequence[EventRecord], labels: Sequence[LabelRecord],
                     ranges: Sequence[QueryRangeRecord]) -> None:
    validate_events(events)
    expected_label_ids = [e.event_id for e in events if e.kind in (3, 4)]
    if [item.event_id for item in labels] != expected_label_ids:
        raise ContractError("labels do not exactly cover candidate/exact-only events")
    covered = 0
    for item in ranges:
        if item.begin_event != covered or item.event_count <= 0:
            raise ContractError("invalid query range")
        end = covered + item.event_count
        if end > len(events):
            raise ContractError("query range exceeds event count")
        if (events[covered].kind != 1 or events[end - 1].kind != 5 or
                events[covered].query_id != item.query_id or
                events[end - 1].query_id != item.query_id):
            raise ContractError("query range boundary mismatch")
        covered = end
    if covered != len(events):
        raise ContractError("query ranges do not cover all events")


def file_entry(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    value = Path(path)
    return {"path": value.name, "size": value.stat().st_size,
            "sha256": sha256_file(value)}


def append_ledger(path: os.PathLike[str] | str, stage: str, status: str,
                  details: Mapping[str, Any]) -> None:
    ledger = Path(path).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "details": dict(details),
    }
    with ledger.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
