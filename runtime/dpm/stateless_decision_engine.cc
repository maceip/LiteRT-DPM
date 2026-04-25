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

#include "runtime/dpm/stateless_decision_engine.h"

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projector.h"
#include "runtime/dpm/event.h"
#include "runtime/dpm/event_sourced_log.h"
#include "runtime/dpm/projection_prompt.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

int64_t SystemDPMClock::NowMicros() const {
  return absl::ToUnixMicros(absl::Now());
}

StatelessDecisionEngine::StatelessDecisionEngine(
    EventSourcedLog* log, DPMProjector* projector,
    DPMInferenceRunner* decider_runner, DPMClock* clock)
    : log_(log),
      projector_(projector),
      decider_runner_(decider_runner),
      clock_(clock == nullptr ? &system_clock_ : clock) {}

absl::StatusOr<DPMDecisionResponse> StatelessDecisionEngine::Decide(
    const DPMDecisionRequest& request, const Config& config) {
  if (log_ == nullptr || projector_ == nullptr || decider_runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM decision engine requires log, projector, and decider runner.");
  }
  if (config.model_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM decision requires a pinned model_id.");
  }
  if (!config.allow_wall_clock_timestamps &&
      (!request.timestamp_us.has_value() ||
       !request.response_timestamp_us.has_value())) {
    return absl::InvalidArgumentError(
        "DPM replay-safe decisions require request timestamp_us and "
        "response_timestamp_us; set allow_wall_clock_timestamps only for "
        "first-write capture.");
  }

  RETURN_IF_ERROR(log_->Append(Event{
      .type = request.type,
      .payload = request.payload,
      .timestamp_us = request.timestamp_us.value_or(clock_->NowMicros()),
  }));

  DPMProjector::ProjectionConfig projection_config = config.projection;
  if (projection_config.model_id.empty()) {
    projection_config.model_id = config.model_id;
  }
  ASSIGN_OR_RETURN(std::string projected_memory,
                   projector_->Project(*log_, projection_config));
  const std::string prompt = CreateDeciderPrompt(
      projected_memory, request.case_id, config.decision_options);
  ASSIGN_OR_RETURN(std::string decision,
                   decider_runner_->Generate(
                       prompt, DPMInferenceConfig{
                                   .max_output_tokens =
                                       config.max_decision_tokens,
                                   .seed = config.seed,
                                   .temperature = config.temperature,
                                   .fresh_context = true,
                                   .model_id = config.model_id,
                               }));
  RETURN_IF_ERROR(log_->Append(Event{
      .type = Event::Type::kModel,
      .payload = decision,
      .timestamp_us =
          request.response_timestamp_us.value_or(clock_->NowMicros()),
      .model_id = config.model_id,
  }));

  return DPMDecisionResponse{
      .projected_memory = projected_memory,
      .decision_text = decision,
  };
}

}  // namespace litert::lm
