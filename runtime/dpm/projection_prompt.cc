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

#include "runtime/dpm/projection_prompt.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/event.h"

namespace litert::lm {
absl::StatusOr<std::string> CreateProjectionPrompt(
    const std::vector<Event>& events, absl::string_view schema_id,
    absl::string_view schema_json, size_t max_event_json_chars) {
  if (schema_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM projection requires a non-empty schema_id.");
  }
  if (schema_json.empty()) {
    return absl::InvalidArgumentError(
        "DPM projection requires a non-empty task schema.");
  }
  const std::string event_json = EventsToJson(events);
  if (max_event_json_chars > 0 && event_json.size() > max_event_json_chars) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "DPM event log is too large for a single projection prompt (",
        event_json.size(), " bytes > ", max_event_json_chars,
        "); hierarchical projection is required."));
  }
  return absl::StrCat(
      "Act as a memory projection engine. Convert the append-only event log "
      "into compact Projected Memory for a stateless decision model.\n\n",
      "Rules:\n",
      "- Preserve Facts, Reasoning, and Compliance as separate fields.\n",
      "- Output valid JSON only.\n",
      "- Use temperature 0 behavior: do not invent facts.\n",
      "- Reference every fact by Event Index using the form [i].\n",
      "- Treat correction events as superseding earlier conflicting facts.\n\n",
      "[SCHEMA ID]\n", schema_id, "\n\n",
      "[TASK SCHEMA]\n", schema_json, "\n\n",
      "[EVENT LOG]\n", event_json, "\n");
}

std::string CreateDeciderPrompt(absl::string_view projected_memory,
                                absl::string_view case_id,
                                absl::string_view decision_options) {
  return absl::StrCat(
      "Given the following Projected Memory M, provide a final verdict for "
      "Case ID ",
      case_id, ".\n",
      "Decision options: ", decision_options, ".\n",
      "Base the decision strictly on Facts and Compliance.\n\n",
      "[PROJECTED MEMORY]\n", projected_memory, "\n");
}

}  // namespace litert::lm
