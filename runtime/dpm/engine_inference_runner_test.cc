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

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "gtest/gtest.h"
#include "runtime/components/tokenizer.h"
#include "runtime/dpm/dpm_projector.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

class FakeSession : public Engine::Session {
 public:
  explicit FakeSession(int initial_step, SessionConfig session_config)
      : current_step_(initial_step),
        session_config_(std::move(session_config)) {}

  absl::StatusOr<Responses> GenerateContent(
      const std::vector<InputData>& contents) override {
    current_step_ += static_cast<int>(contents.size());
    return Responses(TaskState::kDone, {"ok"});
  }

  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override {
    callback(GenerateContent(contents));
    return absl::OkStatus();
  }

  absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) override {
    (void)decode_config;
    callback(GenerateContent(contents));
    return absl::OkStatus();
  }

  absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths) override {
    (void)target_text;
    (void)store_token_lengths;
    return absl::UnimplementedError("not used");
  }

  absl::Status RunPrefill(const std::vector<InputData>& contents) override {
    current_step_ += static_cast<int>(contents.size());
    return absl::OkStatus();
  }

  absl::StatusOr<Responses> RunDecode() override {
    return Responses(TaskState::kDone, {"ok"});
  }

  absl::StatusOr<Responses> RunDecode(
      const DecodeConfig& decode_config) override {
    (void)decode_config;
    return Responses(TaskState::kDone, {"ok"});
  }

  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override {
    return absl::UnimplementedError("not used");
  }

  absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() override {
    return absl::UnimplementedError("not used");
  }

  absl::Status WaitUntilDone() override { return absl::OkStatus(); }

  absl::StatusOr<int> GetCurrentStep() const override { return current_step_; }

  const SessionConfig& GetSessionConfig() const override {
    return session_config_;
  }

 private:
  int current_step_ = 0;
  SessionConfig session_config_;
};

class FakeEngine : public Engine {
 public:
  explicit FakeEngine(int initial_step) : initial_step_(initial_step) {}

  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override {
    ++create_session_calls;
    last_session_config = session_config;
    return std::make_unique<FakeSession>(initial_step_, session_config);
  }

  absl::Status WaitUntilDone(absl::Duration timeout) override {
    (void)timeout;
    return absl::OkStatus();
  }

  const EngineSettings& GetEngineSettings() const override {
    ABSL_LOG(FATAL) << "not used";
    const EngineSettings* settings = nullptr;
    return *settings;
  }

  const Tokenizer& GetTokenizer() const override {
    ABSL_LOG(FATAL) << "not used";
    const Tokenizer* tokenizer = nullptr;
    return *tokenizer;
  }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return absl::UnimplementedError("not used");
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return absl::UnimplementedError("not used");
  }

  int create_session_calls = 0;
  SessionConfig last_session_config = SessionConfig::CreateDefault();

 private:
  int initial_step_ = 0;
};

TEST(EngineDPMInferenceRunnerTest, RejectsNonFreshContextConfig) {
  FakeEngine engine(0);
  EngineDPMInferenceRunner runner(&engine, SessionConfig::CreateDefault());

  EXPECT_FALSE(runner.Generate("prompt", DPMInferenceConfig{
                                             .fresh_context = false,
                                         }).ok());
  EXPECT_EQ(engine.create_session_calls, 0);
}

TEST(EngineDPMInferenceRunnerTest, RejectsMissingModelId) {
  FakeEngine engine(0);
  EngineDPMInferenceRunner runner(&engine, SessionConfig::CreateDefault());

  EXPECT_FALSE(runner.Generate("prompt", DPMInferenceConfig{}).ok());
  EXPECT_EQ(engine.create_session_calls, 0);
}

TEST(EngineDPMInferenceRunnerTest, RejectsSessionWithLeakedStep) {
  FakeEngine engine(7);
  EngineDPMInferenceRunner runner(&engine, SessionConfig::CreateDefault());

  EXPECT_FALSE(runner.Generate("prompt", DPMInferenceConfig{
                                             .model_id = "pinned-test-model",
                                         }).ok());
  EXPECT_EQ(engine.create_session_calls, 1);
}

TEST(EngineDPMInferenceRunnerTest, CreatesFreshSessionWithDeterministicSampler) {
  FakeEngine engine(0);
  EngineDPMInferenceRunner runner(&engine, SessionConfig::CreateDefault());

  ASSERT_OK_AND_ASSIGN(std::string output,
                       runner.Generate("prompt", DPMInferenceConfig{
                                                     .max_output_tokens = 5,
                                                     .seed = 123,
                                                     .temperature = 0.0f,
                                                     .fresh_context = true,
                                                     .model_id =
                                                         "pinned-test-model",
                                                 }));

  EXPECT_EQ(output, "ok");
  EXPECT_EQ(engine.create_session_calls, 1);
  EXPECT_EQ(engine.last_session_config.GetMaxOutputTokens(), 5);
  EXPECT_EQ(engine.last_session_config.GetSamplerParams().k(), 1);
  EXPECT_EQ(engine.last_session_config.GetSamplerParams().seed(), 123);
  EXPECT_EQ(engine.last_session_config.GetSamplerParams().temperature(), 0.0f);
}

}  // namespace
}  // namespace litert::lm
