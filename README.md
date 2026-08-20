# LiteRT-DPM

LiteRT-DPM is a C++ reference implementation of the three-phase
**Deterministic Projection Memory** (DPM) runtime built on
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM). It preserves the
core DPM work from the original repository before that project became the
[VET sidecar](https://github.com/maceip/vet).

DPM treats an append-only event log as durable agent state. At a decision
boundary, the runtime constructs a deterministic, schema-anchored projection
from that log instead of incrementally mutating a rolling summary. The design
comes from [arXiv:2604.20158](https://arxiv.org/abs/2604.20158).

## Repository layout

The `main` branch contains the completed three-phase runtime substrate:

1. **Stateless projection and replay**
   - append-only, identity-scoped event logs;
   - projection and decision prompts;
   - forced KV reset before prefill;
   - proto-text configuration and deterministic replay tests.
2. **Hierarchical checkpoint substrate**
   - canonical checkpoint manifests and content-addressed local stores;
   - SHA-256 and BLAKE3 hashing;
   - Merkle-DAG provenance, copy-on-write event-log branches, checkpoint
     policies, payload codecs, upload sessions, and replay-safe KV transport
     gates.
3. **Audit, correction, and replay**
   - audit certificates and a durable local audit ledger;
   - checkpoint admission gates and checkpointed projections;
   - projection comparison, replay auditing, correction events, and
     correction-aware replay.

The `feature/memory-catalog` branch contains a later 12-commit runtime upgrade
bundle. “Memory catalog” is the project shorthand for its active-evidence and
correction-aware lookup layer; it is not a separate catalog service. The
branch adds precomputed `ActiveEvidenceView` support, compiled correction
directives, projection memory budgets, replay metadata, Phase 3 runtime policy,
audit-certificate lookup, fork-bounded event reads, range/correction indexes,
and performance smoke coverage.

Later CLI-agent hooks, benchmark-result integrations, VET branding, the VET
binary, release workflows, and verifiable-handoff work are deliberately not
part of this repository.

## Status and boundaries

This is a runtime substrate, not a turnkey cloud deployment.

- The local filesystem event-log, checkpoint, provenance, and audit backends
  are implemented and tested.
- Cloud and hardware providers such as S3 Express, MemoryDB, RDMA/RoCE, and
  vendor-specific NPU checkpoint thaw remain integration work.
- Deterministic substrate behavior does not by itself guarantee identical
  model output. End-to-end determinism also requires a pinned model and
  artifacts, deterministic kernels, temperature `0`, a fixed seed, and an
  inference backend that honors those settings.
- Bazel is the authoritative build surface for all three phases. The inherited
  CMake target currently exposes the Phase 1 DPM library and configuration
  loader only.

See the phase closure notes for exact detail:

- [`runtime/dpm/PHASE1_STATUS.md`](runtime/dpm/PHASE1_STATUS.md)
- [`runtime/dpm/PHASE2_STATUS.md`](runtime/dpm/PHASE2_STATUS.md)
- [`runtime/dpm/PHASE3_STATUS.md`](runtime/dpm/PHASE3_STATUS.md)

## Build and test

Install the upstream LiteRT-LM prerequisites, Bazelisk, and Git LFS. Then:

```bash
git clone https://github.com/maceip/LiteRT-DPM.git
cd LiteRT-DPM
git lfs pull

bazelisk test \
  //runtime/dpm/... \
  //runtime/platform/audit/... \
  //runtime/platform/checkpoint/... \
  //runtime/platform/eventlog/... \
  //runtime/platform/hash/... \
  //runtime/platform/provenance/...
```

The optional pinned-model determinism test is enabled only when
`LITERTLM_ENABLE_E2E_DETERMINISM_TEST=1` and `DPM_DETERMINISM_MODEL_PATH`
points to a pinned model artifact.

To inspect the catalog/index upgrade without changing `main`:

```bash
git switch feature/memory-catalog
```

## License and attribution

The repository retains the upstream Apache License 2.0 and source-file
attribution. LiteRT-LM is maintained by Google. The DPM architecture is from
Srinivasan et al., [arXiv:2604.20158](https://arxiv.org/abs/2604.20158).
