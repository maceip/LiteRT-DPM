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

#include "runtime/dpm/engine_inference_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projector.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

EngineDPMInferenceRunner::EngineDPMInferenceRunner(
    Engine* engine, SessionConfig base_session_config)
    : engine_(engine), base_session_config_(std::move(base_session_config)) {}

absl::StatusOr<std::string> EngineDPMInferenceRunner::Generate(
    absl::string_view prompt, const DPMInferenceConfig& config) {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("DPM engine runner has null engine.");
  }
  if (!config.fresh_context) {
    return absl::InvalidArgumentError(
        "DPM Phase 1 requires fresh_context=true for every inference.");
  }
  if (config.model_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM Phase 1 requires a pinned model_id for every inference.");
  }

  SessionConfig session_config = base_session_config_;
  session_config.GetMutableSamplerParams() = CreateDPMSamplerParameters(config);
  session_config.SetMaxOutputTokens(config.max_output_tokens);
  // Phase 1 statelessness: every Prefill must start from a zeroed KV cache.
  // The executor reads this flag and zeros buffers before prefill.
  session_config.SetForceKvResetBeforePrefill(true);

  ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                   engine_->CreateSession(session_config));
  // New sessions are the fresh-context boundary. Backends that expose a step
  // probe get an additional guard against leaked KV state.
  absl::StatusOr<int> initial_step = session->GetCurrentStep();
  if (initial_step.ok()) {
    if (*initial_step != 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "DPM fresh-context session started with nonzero KV step: ",
          *initial_step));
    }
  } else if (initial_step.status().code() !=
             absl::StatusCode::kUnimplemented) {
    return initial_step.status();
  }
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(std::string(prompt)));
  ASSIGN_OR_RETURN(Responses responses,
                   session->GenerateContent(std::move(inputs)));

  std::string text;
  const std::vector<std::string> texts = responses.GetTexts();
  for (const std::string& response : texts) {
    absl::StrAppend(&text, response);
  }
  session.reset();
  return text;
}

}  // namespace litert::lm
