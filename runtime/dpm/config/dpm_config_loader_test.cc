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

#include "runtime/dpm/config/dpm_config_loader.h"

#include "gtest/gtest.h"
#include "runtime/proto/dpm_config.pb.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

constexpr const char* kValidProtoText = R"pb(
identity {
  tenant_id: "tenant-a"
  session_id: "session-1"
}
projection {
  schema_id: "insurance_liability_v2"
  schema_json: "{\"Facts\":[\"string with [i]\"]}"
  memory_budget_chars: 1338
  model_id: "pinned-test-model"
  seed: 20260420
}
event_sink {
  posix {
    root_path: "/var/lib/dpm/sessions"
  }
}
decision_options: "[Approve, Deny]"
max_decision_tokens: 256
)pb";

TEST(DpmConfigLoaderTest, ParsesValidProtoText) {
  ASSERT_OK_AND_ASSIGN(proto::DpmConfig config,
                       LoadDpmConfigFromText(kValidProtoText));
  EXPECT_EQ(config.identity().tenant_id(), "tenant-a");
  EXPECT_EQ(config.projection().schema_id(), "insurance_liability_v2");
  EXPECT_EQ(config.projection().memory_budget_chars(), 1338);
  EXPECT_EQ(config.event_sink().posix().root_path(),
            "/var/lib/dpm/sessions");
  EXPECT_EQ(config.max_decision_tokens(), 256);
}

TEST(DpmConfigLoaderTest, RejectsMalformedProtoText) {
  EXPECT_FALSE(LoadDpmConfigFromText("this is not proto text").ok());
}

TEST(DpmConfigLoaderTest, ToDPMLogIdentityRejectsEmptyFields) {
  proto::DpmLogIdentity proto;
  EXPECT_FALSE(ToDPMLogIdentity(proto).ok());
  proto.set_tenant_id("tenant-a");
  EXPECT_FALSE(ToDPMLogIdentity(proto).ok());
  proto.set_session_id("session-1");
  EXPECT_OK(ToDPMLogIdentity(proto));
}

TEST(DpmConfigLoaderTest, ToProjectionConfigRequiresSchemaAndModel) {
  proto::DpmProjectionConfig proto;
  EXPECT_FALSE(ToProjectionConfig(proto).ok());
  proto.set_schema_id("s");
  EXPECT_FALSE(ToProjectionConfig(proto).ok());
  proto.set_schema_json("{}");
  EXPECT_FALSE(ToProjectionConfig(proto).ok());
  proto.set_model_id("m");
  EXPECT_FALSE(ToProjectionConfig(proto).ok());  // memory budget still 0
  proto.set_memory_budget_chars(1024);
  ASSERT_OK_AND_ASSIGN(DPMProjector::ProjectionConfig out,
                       ToProjectionConfig(proto));
  EXPECT_EQ(out.schema_id, "s");
  EXPECT_EQ(out.model_id, "m");
  EXPECT_EQ(out.memory_budget_chars, 1024);
}

TEST(DpmConfigLoaderTest, ToStatelessDecisionEngineConfigRoundTrips) {
  ASSERT_OK_AND_ASSIGN(proto::DpmConfig config,
                       LoadDpmConfigFromText(kValidProtoText));
  ASSERT_OK_AND_ASSIGN(StatelessDecisionEngine::Config engine_config,
                       ToStatelessDecisionEngineConfig(config));
  EXPECT_EQ(engine_config.model_id, "pinned-test-model");
  EXPECT_EQ(engine_config.decision_options, "[Approve, Deny]");
  EXPECT_EQ(engine_config.max_decision_tokens, 256);
  EXPECT_EQ(engine_config.projection.schema_id, "insurance_liability_v2");
  EXPECT_EQ(engine_config.projection.memory_budget_chars, 1338);
  EXPECT_EQ(engine_config.projection.seed, 20260420);
}

}  // namespace
}  // namespace litert::lm
