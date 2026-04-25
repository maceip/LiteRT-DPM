# Phase 1 Audit Recovery

> **Phase 1 is complete for stateless replay and paper-aligned DPM runtime
> semantics. Remaining work is performance tuning for long-sequence prefill
> and Phase 2 checkpointing, not Phase 1 correctness.**

For the reviewer-facing closure checklist with the four-bucket scoring
(complete / accepted substitution / deferred perf / not Phase 1 code), see
`PHASE1_STATUS.md`. This file is the historical record of audit findings
that fed into Phase 1 — keep it for context; the status doc is the
authoritative pass/fail surface.

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
- `SessionConfig::SetForceKvResetBeforePrefill(true)` propagates from the
  DPM runner into the serial and threaded execution managers, which call
  `llm_executor->Reset()` before every `Tasks::Prefill`. This is the literal
  Predict-loop KV reset the structure doc asked for; non-DPM callers leave
  the flag false and retain KV reuse across calls.
- `Decide()` requires request and response timestamps by default; wall-clock
  capture is an explicit opt-in, and model events record the pinned `model_id`.
- The default determinism test exercises the real append/read/prompt-builder
  path over 10 replays. `//runtime/dpm:dpm_determinism_e2e_test` adds an
  opt-in pinned-model harness for environments that mount a deterministic model
  artifact.

## Accepted Phase 1 Substitutions

- The original AWS SDK uploader item is superseded for raw event logs by
  `PosixEventSink` writing to local disk, EFS, or an S3 Files mount. This keeps
  Phase 1 runtime code SDK-free while preserving append-only durability
  semantics. Empirically verified during the April 25, 2026 probe: bucket-
  level Object Lock COMPLIANCE applies to S3 Files-synced objects;
  `fdatasync` is durable; concurrent O_APPEND is atomic; Lambda mounts work.
- S3 Object Lock is a bucket-provisioning property for the S3 Files path.
  Per-session retention overrides are recorded by `PosixEventSink` as a
  `events.dpmlog.retention.json` sidecar driven by `EventSink::RetentionPolicy`
  / `DpmRetentionPolicy` proto. Bucket-level Object Lock remains the
  load-bearing immutability mechanism for synced objects.
- YAML is replaced for Phase 1 by `runtime/proto/dpm_config.proto`. The
  loader at `runtime/dpm/config/dpm_config_loader.{h,cc}` adapts proto-text
  config into the runtime structs and matches the rest of LiteRT-LM's
  configuration story. The runtime does not take a YAML dependency.

## Follow-up Workstreams

- Long-sequence parallel prefill tuning and hardware baselines. The opt-in
  benchmark entry point is
  `//tools/benchmarks/dpm_prefill_bench`.
- Phase 2 checkpoint substrate and any cloud SDK intake for tensor checkpoint
  blobs.
