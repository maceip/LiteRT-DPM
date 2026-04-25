// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_CHECKPOINT_CHECKPOINT_CLOUD_LAYOUT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_CHECKPOINT_CHECKPOINT_CLOUD_LAYOUT_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Provider-neutral layout knobs for the production checkpoint split:
// checkpoint blobs go to an S3 Express One Zone directory bucket; hot
// metadata, locks, and latest pointers go to MemoryDB; a compact JSONL
// manifest sidecar is written under an Athena-queryable prefix.
struct S3ExpressCheckpointLayout {
  std::string directory_bucket;
  std::string region;
  std::string availability_zone_id;
  std::string blob_prefix = "dpm/checkpoints";

  std::string memorydb_endpoint;
  std::string memorydb_metadata_prefix = "dpm:checkpoint";

  std::string athena_manifest_bucket;
  std::string athena_manifest_prefix = "dpm/checkpoint_manifests";
};

struct CheckpointCloudPlacement {
  std::string blob_object_key;
  std::string blob_uri;
  std::string metadata_uri;
  std::string athena_manifest_object_key;
  std::string athena_manifest_uri;
};

// Builds deterministic object names and URIs for a checkpoint manifest.
// branch_id may be empty for the primary line; it is rendered as "main"
// in partition keys so Athena queries do not need NULL handling.
absl::StatusOr<CheckpointCloudPlacement> BuildS3ExpressCheckpointPlacement(
    const S3ExpressCheckpointLayout& layout, absl::string_view tenant_id,
    absl::string_view session_id, absl::string_view branch_id,
    const Hash256& manifest_hash);

struct AthenaCheckpointRecord {
  std::string tenant_id;
  std::string session_id;
  std::string branch_id;
  Hash256 manifest_hash;
  Hash256 body_hash;
  std::string blob_uri;
  std::string metadata_uri;
  uint64_t base_event_index = 0;
  uint32_t level = 0;
  int64_t created_unix_micros = 0;
  std::string model_id;
  std::string architecture_tag;
  std::string kv_dtype;
};

// Deterministic compact JSONL row for Athena ingestion. The row intentionally
// duplicates blob_uri and metadata_uri so audit queries do not need to join
// against the hot metadata store.
absl::StatusOr<std::string> RenderAthenaCheckpointRecordJson(
    const AthenaCheckpointRecord& record);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_CHECKPOINT_CHECKPOINT_CLOUD_LAYOUT_H_
