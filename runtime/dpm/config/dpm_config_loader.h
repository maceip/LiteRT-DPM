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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CONFIG_DPM_CONFIG_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CONFIG_DPM_CONFIG_LOADER_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projector.h"
#include "runtime/dpm/event_sourced_log.h"
#include "runtime/dpm/stateless_decision_engine.h"
#include "runtime/platform/eventlog/event_sink.h"
#include "runtime/proto/dpm_config.pb.h"

namespace litert::lm {

// Parses a proto-text DpmConfig. Mirrors the rest of LiteRT-LM's
// configuration story (proto-text source of truth, no external YAML
// dependency in the runtime).
absl::StatusOr<proto::DpmConfig> LoadDpmConfigFromText(
    absl::string_view proto_text);

// Reads proto-text from a path and parses it.
absl::StatusOr<proto::DpmConfig> LoadDpmConfigFromFile(absl::string_view path);

// Adapters from the proto into the C++ structs the runtime already consumes.
// Each performs the same validation that StatelessDecisionEngine and
// DPMProjector enforce at call time, so loader-time misconfiguration is
// caught up front.
absl::StatusOr<DPMLogIdentity> ToDPMLogIdentity(
    const proto::DpmLogIdentity& proto);
absl::StatusOr<DPMProjector::ProjectionConfig> ToProjectionConfig(
    const proto::DpmProjectionConfig& proto);
absl::StatusOr<StatelessDecisionEngine::Config> ToStatelessDecisionEngineConfig(
    const proto::DpmConfig& proto);
EventSink::RetentionPolicy ToRetentionPolicy(
    const proto::DpmRetentionPolicy& proto);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CONFIG_DPM_CONFIG_LOADER_H_
