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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_EVENT_SINK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_EVENT_SINK_H_

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Substrate-level append-only record store. Implementations own the on-disk
// (or remote) framing, locking, and durability semantics. Higher layers
// (e.g. DPM's EventSourcedLog) supply opaque record bytes and consume them
// back in append order.
//
// AppendRecord must return only after the record is durable on the substrate.
class EventSink {
 public:
  virtual ~EventSink() = default;

  virtual absl::Status AppendRecord(absl::string_view tenant_id,
                                    absl::string_view session_id,
                                    absl::string_view record_payload) = 0;

  virtual absl::StatusOr<std::vector<std::string>> ReadRecords(
      absl::string_view tenant_id, absl::string_view session_id) const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_EVENT_SINK_H_
