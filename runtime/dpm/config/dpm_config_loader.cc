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

#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "google/protobuf/text_format.h"
#include "runtime/dpm/dpm_projector.h"
#include "runtime/dpm/event_sourced_log.h"
#include "runtime/dpm/stateless_decision_engine.h"
#include "runtime/platform/eventlog/event_sink.h"
#include "runtime/proto/dpm_config.pb.h"

namespace litert::lm {

absl::StatusOr<proto::DpmConfig> LoadDpmConfigFromText(
    absl::string_view proto_text) {
  proto::DpmConfig config;
  if (!google::protobuf::TextFormat::ParseFromString(std::string(proto_text),
                                                     &config)) {
    return absl::InvalidArgumentError(
        "DpmConfig proto-text failed to parse.");
  }
  return config;
}

absl::StatusOr<proto::DpmConfig> LoadDpmConfigFromFile(
    absl::string_view path) {
  std::ifstream file{std::string(path)};
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("DpmConfig file not found: ", path));
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return LoadDpmConfigFromText(buffer.str());
}

absl::StatusOr<DPMLogIdentity> ToDPMLogIdentity(
    const proto::DpmLogIdentity& proto) {
  if (proto.tenant_id().empty()) {
    return absl::InvalidArgumentError(
        "DpmConfig.identity.tenant_id is required.");
  }
  if (proto.session_id().empty()) {
    return absl::InvalidArgumentError(
        "DpmConfig.identity.session_id is required.");
  }
  return DPMLogIdentity{
      .tenant_id = proto.tenant_id(),
      .session_id = proto.session_id(),
  };
}

absl::StatusOr<DPMProjector::ProjectionConfig> ToProjectionConfig(
    const proto::DpmProjectionConfig& proto) {
  if (proto.schema_id().empty()) {
    return absl::InvalidArgumentError(
        "DpmConfig.projection.schema_id is required.");
  }
  if (proto.schema_json().empty()) {
    return absl::InvalidArgumentError(
        "DpmConfig.projection.schema_json is required.");
  }
  if (proto.model_id().empty()) {
    return absl::InvalidArgumentError(
        "DpmConfig.projection.model_id is required (replay-time backend "
        "drift detection depends on it).");
  }
  if (proto.memory_budget_chars() == 0) {
    return absl::InvalidArgumentError(
        "DpmConfig.projection.memory_budget_chars must be non-zero.");
  }

  DPMProjector::ProjectionConfig out;
  out.schema_id = proto.schema_id();
  out.schema_json = proto.schema_json();
  out.memory_budget_chars = proto.memory_budget_chars();
  if (proto.max_event_log_chars() != 0) {
    out.max_event_log_chars = proto.max_event_log_chars();
  }
  if (proto.max_tokens() != 0) {
    out.max_tokens = proto.max_tokens();
  }
  out.model_id = proto.model_id();
  if (proto.seed() != 0) {
    out.seed = proto.seed();
  }
  return out;
}

EventSink::RetentionPolicy ToRetentionPolicy(
    const proto::DpmRetentionPolicy& proto) {
  EventSink::RetentionPolicy out;
  out.retain_until_unix_seconds = proto.retain_until_unix_seconds();
  out.legal_hold = proto.legal_hold();
  return out;
}

absl::StatusOr<StatelessDecisionEngine::Config>
ToStatelessDecisionEngineConfig(const proto::DpmConfig& proto) {
  if (!proto.has_projection()) {
    return absl::InvalidArgumentError(
        "DpmConfig.projection is required.");
  }
  StatelessDecisionEngine::Config out;
  auto projection = ToProjectionConfig(proto.projection());
  if (!projection.ok()) {
    return projection.status();
  }
  out.projection = *projection;
  out.model_id = proto.projection().model_id();
  if (!proto.decision_options().empty()) {
    out.decision_options = proto.decision_options();
  }
  if (proto.max_decision_tokens() != 0) {
    out.max_decision_tokens = proto.max_decision_tokens();
  }
  return out;
}

}  // namespace litert::lm
