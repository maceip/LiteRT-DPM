# Contributing

LiteRT-DPM is scoped to the three-phase Deterministic Projection Memory
runtime and its tests. Please open an issue in this repository before a large
change so the intended phase boundary and compatibility impact are clear.

For code changes:

1. Keep the DPM core independent of cloud-provider SDKs. Provider-specific
   implementations should sit behind the existing storage, event-log,
   checkpoint, audit, or inference interfaces.
2. Add focused Bazel tests for changed behavior.
3. Run the affected `//runtime/dpm/...` and `//runtime/platform/...` tests.
4. Do not commit credentials, private session logs, model files, generated
   benchmark runs, or other user data.

Issues in upstream LiteRT-LM itself should be reported to
[google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/issues).
