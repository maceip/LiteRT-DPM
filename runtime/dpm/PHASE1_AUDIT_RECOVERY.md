# Phase 1 Audit Recovery

This file tracks audit findings that are intentionally part of the Phase 1
definition of done, so omissions do not silently roll into Phase 2.

## Closed In Runtime

- Event log records are tenant/session scoped by construction.
- Durability + framing is owned by `runtime/platform/eventlog/PosixEventSink`,
  not by DPM directly. Appends use inter-process file locking and
  fsync/FlushFileBuffers; on-disk records are length-prefixed so partial
  writes are detected; reads use `MemoryMappedFile`.
- `EventSourcedLog` is now a decoded-event facade over an injectable
  `EventSink`; the legacy root-path constructor builds an owned
  `PosixEventSink` for backwards compatibility.
- `GetAllEvents()` and projection prompt construction keep in-memory caches
  keyed by `EventSink::ProbeGeneration()` when the substrate has a cheap
  generation token; sinks without a probe fall back to an Abseil content
  fingerprint.
- Projection prompt construction streams paper-style `[i] event` lines from the
  event sink instead of rebuilding a pretty JSON array from decoded events.
- Projection prompts fail instead of truncating oversized logs.
- Projection prompts require a non-empty schema id and JSON schema.
- Projection outputs must parse as JSON and include `[i]` citations in
  `Facts`, `Reasoning`, and `Compliance`.
- Fresh-context inference rejects `fresh_context=false`; backends with a
  current-step probe must start new sessions at KV step 0.
- `Decide()` requires request and response timestamps by default; wall-clock
  capture is an explicit opt-in, and model events record the pinned `model_id`.

## Still Open Before Claiming Full Paper Parity

- AWS SDK for C++ multipart uploader on a detached background thread, gated by
  `remote_sync.enabled`.
- S3 Object Lock metadata plumbing, including `CreatedAt` and `RetainUntil`
  upload metadata hints.
- YAML config wiring for DPM projection, log identity, and remote sync.
- XNNPack / ML Drift long-sequence parallel prefill tuning and benchmarks.
