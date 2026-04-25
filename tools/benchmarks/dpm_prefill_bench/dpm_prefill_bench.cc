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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/dpm/projection_prompt.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, model_path, "", "Pinned .litertlm model path.");
ABSL_FLAG(std::string, backend, "cpu",
          "Backend to measure: cpu, gpu, npu, cpu_artisan, gpu_artisan.");
ABSL_FLAG(int, trajectory_chars, 27000,
          "Approximate event-log character count before prompt framing.");
ABSL_FLAG(int, memory_budget_chars, 5352,
          "DPM projection memory budget passed to the prompt.");
ABSL_FLAG(int, max_event_log_chars, 131072,
          "Maximum event-log bytes accepted by prompt construction.");
ABSL_FLAG(int, max_num_tokens, 32768,
          "Engine KV-cache capacity for the benchmark session.");
ABSL_FLAG(int, iterations, 1,
          "Number of fresh sessions to prefill with the same DPM prompt.");
ABSL_FLAG(int, num_cpu_threads, 0,
          "CPU/XNNPack thread count override. 0 keeps the model default.");
ABSL_FLAG(int, prefill_chunk_size, -1,
          "CPU dynamic-prefill chunk size. -1 keeps the model default.");
ABSL_FLAG(std::string, cache_dir, "",
          "Delegate cache directory. Empty keeps the runtime default.");
ABSL_FLAG(bool, disable_cache, false,
          "Disable delegate caches by setting the cache dir to :nocache.");
ABSL_FLAG(bool, clear_kv_cache_before_prefill, true,
          "Clear KV buffers before first prefill.");
ABSL_FLAG(std::string, schema_id, "insurance_liability_v2",
          "Projection schema id used to frame the benchmark prompt.");
ABSL_FLAG(
    std::string, schema_json,
    R"json({"Facts":["string with one-based [i] citation"],)"
    R"json("Reasoning":["string with one-based [i] citation"],)"
    R"json("Compliance":["string with one-based [i] citation"]})json",
    "Projection schema JSON used to frame the benchmark prompt.");
ABSL_FLAG(std::string, output_json, "",
          "Optional path for machine-readable benchmark results.");

namespace {

using ::litert::lm::AdvancedSettings;
using ::litert::lm::Backend;
using ::litert::lm::BenchmarkInfo;
using ::litert::lm::CpuConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineFactory;
using ::litert::lm::EngineSettings;
using ::litert::lm::GetBackendFromString;
using ::litert::lm::GetBackendString;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::ModelAssets;
using ::litert::lm::SessionConfig;
using ::nlohmann::ordered_json;

std::string MakeSyntheticEventLog(int target_chars) {
  std::string event_log;
  int index = 1;
  while (event_log.size() < static_cast<size_t>(target_chars)) {
    ordered_json event = {
        {"type", index % 7 == 0 ? "correction" : "tool"},
        {"tenant_id", "bench-tenant"},
        {"session_id", "bench-session"},
        {"timestamp_us", 1000000 + index},
        {"payload",
         absl::StrCat(
             "CASE-", index,
             " reports policy limit $100000, deductible $500, loss date "
             "2026-04-03, invoice INV-",
             7000 + index,
             ", and compliance anchor Regulation B section 1002.",
             index % 7 == 0
                 ? " Correction: keep dollar anchors verbatim."
                 : "")},
    };
    absl::StrAppend(&event_log, "[", index, "] ", event.dump(), "\n");
    ++index;
  }
  return event_log;
}

absl::StatusOr<EngineSettings> CreateBenchmarkEngineSettings(
    absl::string_view model_path, Backend backend) {
  ASSIGN_OR_RETURN(ModelAssets model_assets, ModelAssets::Create(model_path));
  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   EngineSettings::CreateDefault(std::move(model_assets),
                                                 backend));

  auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
  if (absl::GetFlag(FLAGS_max_num_tokens) > 0) {
    executor_settings.SetMaxNumTokens(absl::GetFlag(FLAGS_max_num_tokens));
  }
  if (absl::GetFlag(FLAGS_disable_cache)) {
    executor_settings.SetCacheDir(":nocache");
  } else if (!absl::GetFlag(FLAGS_cache_dir).empty()) {
    executor_settings.SetCacheDir(absl::GetFlag(FLAGS_cache_dir));
  }

  if (backend == Backend::CPU) {
    ASSIGN_OR_RETURN(CpuConfig cpu_config,
                     executor_settings.MutableBackendConfig<CpuConfig>());
    if (absl::GetFlag(FLAGS_num_cpu_threads) > 0) {
      cpu_config.number_of_threads =
          static_cast<uint32_t>(absl::GetFlag(FLAGS_num_cpu_threads));
    }
    cpu_config.prefill_chunk_size = absl::GetFlag(FLAGS_prefill_chunk_size);
    executor_settings.SetBackendConfig(cpu_config);
  }

  AdvancedSettings advanced_settings;
  if (executor_settings.GetAdvancedSettings().has_value()) {
    advanced_settings = *executor_settings.GetAdvancedSettings();
  }
  advanced_settings.clear_kv_cache_before_prefill =
      absl::GetFlag(FLAGS_clear_kv_cache_before_prefill);
  advanced_settings.is_benchmark = true;
  executor_settings.SetAdvancedSettings(advanced_settings);

  // Enables BenchmarkInfo collection for the explicit RunPrefill calls below.
  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();
  return engine_settings;
}

ordered_json BenchmarkInfoToJson(const BenchmarkInfo& info) {
  ordered_json json = ordered_json::object();
  json["prefill_turns"] = ordered_json::array();
  for (uint64_t i = 0; i < info.GetTotalPrefillTurns(); ++i) {
    auto turn = info.GetPrefillTurn(static_cast<int>(i));
    if (!turn.ok()) {
      continue;
    }
    json["prefill_turns"].push_back({
        {"index", i},
        {"tokens", turn->num_tokens},
        {"duration_ms", absl::ToDoubleMilliseconds(turn->duration)},
        {"tokens_per_second",
         info.GetPrefillTokensPerSec(static_cast<int>(i))},
    });
  }

  json["text_to_token_ids_turns"] = ordered_json::array();
  for (uint64_t i = 0; i < info.GetTotalTextToTokenIdsTurns(); ++i) {
    auto turn = info.GetTextToTokenIdsTurn(static_cast<int>(i));
    if (!turn.ok()) {
      continue;
    }
    json["text_to_token_ids_turns"].push_back({
        {"index", i},
        {"tokens", turn->num_tokens},
        {"duration_ms", absl::ToDoubleMilliseconds(turn->duration)},
    });
  }
  return json;
}

absl::Status Main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("--model_path is required.");
  }
  if (absl::GetFlag(FLAGS_trajectory_chars) <= 0) {
    return absl::InvalidArgumentError("--trajectory_chars must be positive.");
  }
  if (absl::GetFlag(FLAGS_iterations) <= 0) {
    return absl::InvalidArgumentError("--iterations must be positive.");
  }

  ASSIGN_OR_RETURN(Backend backend,
                   GetBackendFromString(absl::GetFlag(FLAGS_backend)));
  const std::string event_log =
      MakeSyntheticEventLog(absl::GetFlag(FLAGS_trajectory_chars));
  ASSIGN_OR_RETURN(
      std::string prompt,
      litert::lm::CreateProjectionPrompt(
          event_log, absl::GetFlag(FLAGS_schema_id),
          absl::GetFlag(FLAGS_schema_json),
          static_cast<size_t>(absl::GetFlag(FLAGS_memory_budget_chars)),
          static_cast<size_t>(absl::GetFlag(FLAGS_max_event_log_chars))));

  const absl::Time init_start = absl::Now();
  ASSIGN_OR_RETURN(EngineSettings engine_settings,
                   CreateBenchmarkEngineSettings(model_path, backend));
  ASSIGN_OR_RETURN(std::unique_ptr<Engine> engine,
                   EngineFactory::CreateDefault(std::move(engine_settings),
                                                prompt));
  const absl::Duration engine_init_duration = absl::Now() - init_start;

  ordered_json result = {
      {"schema_version", 1},
      {"bench", "dpm_prefill_bench"},
      {"backend", GetBackendString(backend)},
      {"model_path_basename",
       std::filesystem::path(model_path).filename().string()},
      {"trajectory_chars_requested", absl::GetFlag(FLAGS_trajectory_chars)},
      {"event_log_chars", event_log.size()},
      {"prompt_chars", prompt.size()},
      {"memory_budget_chars", absl::GetFlag(FLAGS_memory_budget_chars)},
      {"max_num_tokens", absl::GetFlag(FLAGS_max_num_tokens)},
      {"num_cpu_threads", absl::GetFlag(FLAGS_num_cpu_threads)},
      {"prefill_chunk_size", absl::GetFlag(FLAGS_prefill_chunk_size)},
      {"clear_kv_cache_before_prefill",
       absl::GetFlag(FLAGS_clear_kv_cache_before_prefill)},
      {"engine_init_ms", absl::ToDoubleMilliseconds(engine_init_duration)},
      {"iterations", ordered_json::array()},
  };

  for (int i = 0; i < absl::GetFlag(FLAGS_iterations); ++i) {
    ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                     engine->CreateSession(SessionConfig::CreateDefault()));
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText(prompt));

    const absl::Time prefill_start = absl::Now();
    RETURN_IF_ERROR(session->RunPrefill(std::move(inputs)));
    const absl::Duration wall_duration = absl::Now() - prefill_start;
    ASSIGN_OR_RETURN(BenchmarkInfo info, session->GetBenchmarkInfo());

    ordered_json iteration = BenchmarkInfoToJson(info);
    iteration["iteration"] = i;
    iteration["wall_ms"] = absl::ToDoubleMilliseconds(wall_duration);
    result["iterations"].push_back(std::move(iteration));
  }

  const std::string rendered = result.dump(2);
  if (absl::GetFlag(FLAGS_output_json).empty()) {
    std::cout << rendered << std::endl;
  } else {
    const std::filesystem::path output_path(absl::GetFlag(FLAGS_output_json));
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    if (!output.is_open()) {
      return absl::UnavailableError(
          absl::StrCat("Could not open output file: ", output_path.string()));
    }
    output << rendered << "\n";
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  const absl::Status status = Main(argc, argv);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}
