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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_STATELESS_DECISION_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_STATELESS_DECISION_ENGINE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projector.h"
#include "runtime/dpm/event.h"
#include "runtime/dpm/event_sourced_log.h"

namespace litert::lm {

class DPMClock {
 public:
  virtual ~DPMClock() = default;
  virtual int64_t NowMicros() const = 0;
};

class SystemDPMClock : public DPMClock {
 public:
  int64_t NowMicros() const override;
};

struct DPMDecisionRequest {
  std::string payload;
  std::string case_id;
  Event::Type type = Event::Type::kUser;
  std::optional<int64_t> timestamp_us = std::nullopt;
  std::optional<int64_t> response_timestamp_us = std::nullopt;
};

struct DPMDecisionResponse {
  std::string projected_memory;
  std::string decision_text;
};

class StatelessDecisionEngine {
 public:
  struct Config {
    DPMProjector::ProjectionConfig projection;
    std::string decision_options = "[Approve, Deny, Request More Info]";
    int max_decision_tokens = 512;
    int seed = 42;
    float temperature = 0.0f;
    std::string model_id;
    bool allow_wall_clock_timestamps = false;
  };

  StatelessDecisionEngine(EventSourcedLog* log, DPMProjector* projector,
                          DPMInferenceRunner* decider_runner,
                          DPMClock* clock = nullptr);

  absl::StatusOr<DPMDecisionResponse> Decide(
      const DPMDecisionRequest& request, const Config& config);

 private:
  EventSourcedLog* log_;
  DPMProjector* projector_;
  DPMInferenceRunner* decider_runner_;
  DPMClock* clock_;
  SystemDPMClock system_clock_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_STATELESS_DECISION_ENGINE_H_
