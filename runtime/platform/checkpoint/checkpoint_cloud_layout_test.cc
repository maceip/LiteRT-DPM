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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

using ::testing::HasSubstr;

Hash256 FilledHash(uint8_t value) {
  Hash256 hash;
  hash.bytes.fill(value);
  return hash;
}

S3ExpressCheckpointLayout Layout() {
  return S3ExpressCheckpointLayout{
      .directory_bucket = "dpm-dir-bucket--use1-az1--x-s3",
      .region = "us-east-1",
      .availability_zone_id = "use1-az1",
      .blob_prefix = "/dpm/checkpoints/",
      .memorydb_endpoint = "clustercfg.dpm.example:6379/",
      .memorydb_metadata_prefix = "/dpm:checkpoint/",
      .athena_manifest_bucket = "dpm-audit-manifests",
      .athena_manifest_prefix = "/dpm/checkpoint_manifests/",
  };
}

TEST(CheckpointCloudLayoutTest, BuildsTieredS3ExpressPlacement) {
  ASSERT_OK_AND_ASSIGN(
      CheckpointCloudPlacement placement,
      BuildS3ExpressCheckpointPlacement(Layout(), "tenant-a", "session-1", "",
                                        FilledHash(0xab)));

  EXPECT_THAT(placement.blob_uri,
              HasSubstr("s3express://dpm-dir-bucket--use1-az1--x-s3/"));
  EXPECT_THAT(placement.blob_object_key,
              HasSubstr("tenant_id=tenant-a/session_id=session-1/"
                        "branch_id=main/"));
  EXPECT_THAT(placement.blob_object_key, HasSubstr(".dpmckpt"));
  EXPECT_THAT(placement.metadata_uri,
              HasSubstr("memorydb://clustercfg.dpm.example:6379/"));
  EXPECT_THAT(placement.athena_manifest_uri,
              HasSubstr("s3://dpm-audit-manifests/"));
  EXPECT_THAT(placement.athena_manifest_object_key, HasSubstr(".jsonl"));
}

TEST(CheckpointCloudLayoutTest, RejectsPartitionUnsafeIdentity) {
  auto placement = BuildS3ExpressCheckpointPlacement(
      Layout(), "tenant/a", "session-1", "", FilledHash(0xab));
  EXPECT_FALSE(placement.ok());
  EXPECT_EQ(placement.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(CheckpointCloudLayoutTest, RendersDeterministicAthenaJsonlRecord) {
  ASSERT_OK_AND_ASSIGN(
      CheckpointCloudPlacement placement,
      BuildS3ExpressCheckpointPlacement(Layout(), "tenant-a", "session-1",
                                        "branch-9", FilledHash(0xab)));

  ASSERT_OK_AND_ASSIGN(
      std::string json,
      RenderAthenaCheckpointRecordJson(AthenaCheckpointRecord{
          .tenant_id = "tenant-a",
          .session_id = "session-1",
          .branch_id = "branch-9",
          .manifest_hash = FilledHash(0xab),
          .body_hash = FilledHash(0xcd),
          .blob_uri = placement.blob_uri,
          .metadata_uri = placement.metadata_uri,
          .base_event_index = 42,
          .level = 1,
          .created_unix_micros = 123456,
          .model_id = "litert-local-model",
          .architecture_tag = "x86_64-xnnpack-fp16",
          .kv_dtype = "fp16",
      }));

  EXPECT_THAT(json, HasSubstr("\"tenant_id\":\"tenant-a\""));
  EXPECT_THAT(json, HasSubstr("\"branch_id\":\"branch-9\""));
  EXPECT_THAT(json, HasSubstr("\"base_event_index\":42"));
  EXPECT_THAT(json, HasSubstr("\"level\":1"));
  EXPECT_THAT(json, HasSubstr("\"kv_dtype\":\"fp16\""));
  EXPECT_EQ(json.back(), '\n');
}

TEST(CheckpointCloudLayoutTest, EscapesJsonFields) {
  ASSERT_OK_AND_ASSIGN(
      std::string json,
      RenderAthenaCheckpointRecordJson(AthenaCheckpointRecord{
          .tenant_id = "tenant-a",
          .session_id = "session-1",
          .branch_id = "",
          .manifest_hash = FilledHash(0xab),
          .body_hash = FilledHash(0xcd),
          .blob_uri = "s3express://bucket/key",
          .metadata_uri = "memorydb://cluster/key",
          .base_event_index = 1,
          .level = 0,
          .created_unix_micros = 2,
          .model_id = "model\"withquote",
          .architecture_tag = "x86_64\nxnnpack",
          .kv_dtype = "fp16",
      }));
  EXPECT_THAT(json, HasSubstr("model\\\"withquote"));
  EXPECT_THAT(json, HasSubstr("x86_64\\nxnnpack"));
  EXPECT_THAT(json, HasSubstr("\"branch_id\":\"main\""));
}

}  // namespace
}  // namespace litert::lm
