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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_INFERENCE_RUNNER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_INFERENCE_RUNNER_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projector.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"

namespace litert::lm {

// DPM runner that enforces fresh-context inference by creating a new session for
// every generation call.
class EngineDPMInferenceRunner : public DPMInferenceRunner {
 public:
  EngineDPMInferenceRunner(Engine* engine, SessionConfig base_session_config);

  absl::StatusOr<std::string> Generate(
      absl::string_view prompt, const DPMInferenceConfig& config) override;

 private:
  Engine* engine_;
  SessionConfig base_session_config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_INFERENCE_RUNNER_H_
