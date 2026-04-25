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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
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

  virtual absl::Status ForEachRecord(
      absl::string_view tenant_id, absl::string_view session_id,
      absl::FunctionRef<absl::Status(absl::string_view)> callback) const {
    absl::StatusOr<std::vector<std::string>> records =
        ReadRecords(tenant_id, session_id);
    if (!records.ok()) {
      return records.status();
    }
    for (const std::string& record : *records) {
      absl::Status status = callback(record);
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }

  // Generation token that consumers can use to invalidate decoded-record
  // caches without re-reading the underlying records. Treat the pair as an
  // opaque cache key: implementations may populate record_count when it is
  // cheap, or leave it at zero and advance opaque_token instead.
  // The default implementation returns kUnimplemented; callers must treat
  // that as "always invalidate the cache".
  struct Generation {
    uint64_t record_count = 0;
    uint64_t opaque_token = 0;
  };
  virtual absl::StatusOr<Generation> ProbeGeneration(
      absl::string_view tenant_id, absl::string_view session_id) const {
    (void)tenant_id;
    (void)session_id;
    return absl::UnimplementedError(
        "EventSink::ProbeGeneration is not implemented by this backend.");
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_EVENT_SINK_H_
