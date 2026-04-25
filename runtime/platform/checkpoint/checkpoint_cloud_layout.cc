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

#include "runtime/platform/checkpoint/checkpoint_cloud_layout.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/hasher.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

bool BadPartitionComponent(absl::string_view value) {
  return value.empty() || value == "." || value == ".." ||
         value.find('/') != absl::string_view::npos ||
         value.find('\\') != absl::string_view::npos ||
         value.find('=') != absl::string_view::npos;
}

std::string BranchPartition(absl::string_view branch_id) {
  return branch_id.empty() ? std::string("main") : std::string(branch_id);
}

std::string TrimSlashes(absl::string_view value) {
  while (!value.empty() && value.front() == '/') value.remove_prefix(1);
  while (!value.empty() && value.back() == '/') value.remove_suffix(1);
  return std::string(value);
}

std::string TrimTrailingSlashes(absl::string_view value) {
  while (!value.empty() && value.back() == '/') value.remove_suffix(1);
  return std::string(value);
}

absl::Status ValidateLayout(const S3ExpressCheckpointLayout& layout) {
  if (layout.directory_bucket.empty() ||
      layout.directory_bucket.find('/') != std::string::npos) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint layout requires directory_bucket.");
  }
  if (layout.region.empty()) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint layout requires region.");
  }
  if (layout.availability_zone_id.empty()) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint layout requires availability_zone_id.");
  }
  if (layout.memorydb_endpoint.empty()) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint layout requires memorydb_endpoint.");
  }
  if (layout.athena_manifest_bucket.empty() ||
      layout.athena_manifest_bucket.find('/') != std::string::npos) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint layout requires athena_manifest_bucket.");
  }
  return absl::OkStatus();
}

absl::Status ValidateIdentity(absl::string_view tenant_id,
                              absl::string_view session_id,
                              absl::string_view branch_id) {
  if (BadPartitionComponent(tenant_id)) {
    return absl::InvalidArgumentError("checkpoint cloud layout: bad tenant_id.");
  }
  if (BadPartitionComponent(session_id)) {
    return absl::InvalidArgumentError(
        "checkpoint cloud layout: bad session_id.");
  }
  if (!branch_id.empty() && BadPartitionComponent(branch_id)) {
    return absl::InvalidArgumentError("checkpoint cloud layout: bad branch_id.");
  }
  return absl::OkStatus();
}

void AppendJsonString(absl::string_view value, std::string* out) {
  constexpr char kHex[] = "0123456789abcdef";
  out->push_back('"');
  for (char c : value) {
    const unsigned char uc = static_cast<unsigned char>(c);
    switch (c) {
      case '"':
        out->append("\\\"");
        break;
      case '\\':
        out->append("\\\\");
        break;
      case '\b':
        out->append("\\b");
        break;
      case '\f':
        out->append("\\f");
        break;
      case '\n':
        out->append("\\n");
        break;
      case '\r':
        out->append("\\r");
        break;
      case '\t':
        out->append("\\t");
        break;
      default:
        if (uc < 0x20) {
          out->append("\\u00");
          out->push_back(kHex[(uc >> 4) & 0x0f]);
          out->push_back(kHex[uc & 0x0f]);
        } else {
          out->push_back(c);
        }
        break;
    }
  }
  out->push_back('"');
}

void AppendJsonStringField(absl::string_view name, absl::string_view value,
                           bool first, std::string* out) {
  if (!first) out->push_back(',');
  AppendJsonString(name, out);
  out->push_back(':');
  AppendJsonString(value, out);
}

template <typename T>
void AppendJsonNumberField(absl::string_view name, T value, bool first,
                           std::string* out) {
  if (!first) out->push_back(',');
  AppendJsonString(name, out);
  absl::StrAppend(out, ":", value);
}

absl::Status ValidateRecord(const AthenaCheckpointRecord& record) {
  RETURN_IF_ERROR(ValidateIdentity(record.tenant_id, record.session_id,
                                   record.branch_id));
  if (IsZeroHash(record.manifest_hash)) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires manifest_hash.");
  }
  if (IsZeroHash(record.body_hash)) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires body_hash.");
  }
  if (record.blob_uri.empty()) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires blob_uri.");
  }
  if (record.metadata_uri.empty()) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires metadata_uri.");
  }
  if (record.model_id.empty()) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires model_id.");
  }
  if (record.architecture_tag.empty()) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires architecture_tag.");
  }
  if (record.kv_dtype.empty()) {
    return absl::InvalidArgumentError(
        "Athena checkpoint record requires kv_dtype.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CheckpointCloudPlacement> BuildS3ExpressCheckpointPlacement(
    const S3ExpressCheckpointLayout& layout, absl::string_view tenant_id,
    absl::string_view session_id, absl::string_view branch_id,
    const Hash256& manifest_hash) {
  RETURN_IF_ERROR(ValidateLayout(layout));
  RETURN_IF_ERROR(ValidateIdentity(tenant_id, session_id, branch_id));
  if (IsZeroHash(manifest_hash)) {
    return absl::InvalidArgumentError(
        "S3 Express checkpoint placement requires manifest_hash.");
  }

  const std::string branch = BranchPartition(branch_id);
  const std::string manifest_hex = manifest_hash.ToHex();
  const std::string blob_prefix = TrimSlashes(layout.blob_prefix);
  const std::string athena_prefix = TrimSlashes(layout.athena_manifest_prefix);
  const std::string partition_path =
      absl::StrCat("tenant_id=", tenant_id, "/session_id=", session_id,
                   "/branch_id=", branch, "/");

  CheckpointCloudPlacement placement;
  placement.blob_object_key =
      absl::StrCat(blob_prefix.empty() ? "" : absl::StrCat(blob_prefix, "/"),
                   partition_path, manifest_hex, ".dpmckpt");
  placement.blob_uri = absl::StrCat("s3express://", layout.directory_bucket,
                                    "/", placement.blob_object_key);

  std::string memorydb_endpoint = TrimTrailingSlashes(layout.memorydb_endpoint);
  if (memorydb_endpoint.rfind("memorydb://", 0) != 0) {
    memorydb_endpoint = absl::StrCat("memorydb://", memorydb_endpoint);
  }
  const std::string metadata_prefix =
      TrimSlashes(layout.memorydb_metadata_prefix);
  placement.metadata_uri = absl::StrCat(
      memorydb_endpoint, "/", metadata_prefix.empty() ? "dpm:checkpoint"
                                                      : metadata_prefix,
      "/", partition_path, manifest_hex);

  placement.athena_manifest_object_key =
      absl::StrCat(athena_prefix.empty() ? "" : absl::StrCat(athena_prefix,
                                                             "/"),
                   partition_path, manifest_hex, ".jsonl");
  placement.athena_manifest_uri =
      absl::StrCat("s3://", layout.athena_manifest_bucket, "/",
                   placement.athena_manifest_object_key);
  return placement;
}

absl::StatusOr<std::string> RenderAthenaCheckpointRecordJson(
    const AthenaCheckpointRecord& record) {
  RETURN_IF_ERROR(ValidateRecord(record));

  std::string out;
  out.push_back('{');
  AppendJsonStringField("tenant_id", record.tenant_id, /*first=*/true, &out);
  AppendJsonStringField("session_id", record.session_id, /*first=*/false,
                        &out);
  AppendJsonStringField("branch_id", BranchPartition(record.branch_id),
                        /*first=*/false, &out);
  AppendJsonStringField("manifest_hash", record.manifest_hash.ToHex(),
                        /*first=*/false, &out);
  AppendJsonStringField("body_hash", record.body_hash.ToHex(),
                        /*first=*/false, &out);
  AppendJsonStringField("blob_uri", record.blob_uri, /*first=*/false, &out);
  AppendJsonStringField("metadata_uri", record.metadata_uri, /*first=*/false,
                        &out);
  AppendJsonNumberField("base_event_index", record.base_event_index,
                        /*first=*/false, &out);
  AppendJsonNumberField("level", record.level, /*first=*/false, &out);
  AppendJsonNumberField("created_unix_micros", record.created_unix_micros,
                        /*first=*/false, &out);
  AppendJsonStringField("model_id", record.model_id, /*first=*/false, &out);
  AppendJsonStringField("architecture_tag", record.architecture_tag,
                        /*first=*/false, &out);
  AppendJsonStringField("kv_dtype", record.kv_dtype, /*first=*/false, &out);
  out.push_back('}');
  out.push_back('\n');
  return out;
}

}  // namespace litert::lm
