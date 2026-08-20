# Phase 3 Status

> **The Phase 3 core substrate is complete at this baseline: audit
> certificates, checkpoint admission, replay auditing, correction events, and
> correction-aware replay are implemented. Distributed audit workers and
> provider-specific deployment remain integration work.**

## Complete in code

- Audit certificate representation, signing interface, query surface, and a
  durable local filesystem audit ledger.
- Optional dynamically loaded liboqs signer integration, without a mandatory
  build-time liboqs dependency.
- Checkpointed projection and projection comparison.
- Checkpoint decision gates that admit audited state or require replay.
- Projection replay auditor over the canonical event log and checkpoint DAG.
- Correction protocol and correction-aware replay behavior.
- Phase 3 audit-flow, audited-loader, replay-auditor, comparator, correction,
  and checkpoint-gate test coverage in the Bazel graph.

## Deliberately outside the core baseline

- A hosted Lambda/SQS/GPU audit fleet or any other particular cloud topology.
- A managed audit-ledger provider.
- Deployment-specific key management and signer provisioning.
- Agent-specific CLI hooks and the standalone VET sidecar.

## Later catalog/index branch

The `feature/memory-catalog` branch builds on this baseline with the
active-evidence view, compiled correction directives, budget and runtime-policy
controls, bounded reads, audit-certificate lookup, and range/correction
indexes. Those optimizations are kept off `main` so this branch remains the
clean three-phase implementation boundary.
