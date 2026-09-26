# Unified edge-estimation schemas

Version 1 uses three canonical little-endian files. Every file starts with a
64-byte header: 8-byte magic, `u32` schema/header/record/dimension fields,
`u64` record count, and a 32-byte edge-catalog identity.

- `events.bin` (`UQEV0001`, 80-byte records) is the ordered input stream and
  never contains exact candidate labels.
- `labels.bin` (`UQLB0001`, 16-byte records) maps candidate/exact-only event
  IDs to the operational exact squared distance.
- `query_ranges.bin` (`UQQR0001`, 24-byte records) partitions the complete
  stream into contiguous original query IDs.

Readers reject unknown versions, record-size changes, truncation/trailing
bytes, non-zero reserved fields, non-contiguous IDs, illegal state transitions,
missing/duplicate labels, and incomplete query-range coverage. Legacy sparse
traces are quality-only and never acquire `ordered_timing=true` by import.

The exact byte offsets are implemented once in
`tools/edge_estimation/event_format.h` and mirrored by
`scripts/edge_estimation/contracts.py`; the round-trip tests cover both sides.
