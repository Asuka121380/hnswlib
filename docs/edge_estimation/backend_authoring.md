# Backend authoring contract

A backend owns immutable model/record storage and thread-private query/source
state. It prepares query state once, source state at the declared eager/lazy
boundary, and returns either a finite edge score or an explicit fallback
status. Configuration dispatch happens outside the per-edge loop.

New dot codecs estimate `(q-c)^T u` and use the shared distance bridge. Legacy
PQ and PQ+QJL use score-level adapters so their production float LUT,
expression order, residual scale, and offset are not rewritten. Invalid
records remain in the common decision denominator and request exact fallback.

Each backend must ship a deterministic tiny fixture, independent reference,
native load/estimate test, layout/byte accounting, artifact/catalog identity
check, and capability declaration. Packed PQ records use real bit packing;
`packed4` stores two subcodes per byte and requires a zero high tail nibble for
odd `M`.
