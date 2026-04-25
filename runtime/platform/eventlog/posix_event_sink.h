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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_POSIX_EVENT_SINK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_POSIX_EVENT_SINK_H_

#include <filesystem>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/eventlog/event_sink.h"

namespace litert::lm {

// EventSink backed by a POSIX-style filesystem rooted at root_path. Records
// for (tenant_id, session_id) are stored in
//   root_path / tenant_id / session_id / events.dpmlog
// using a magic header followed by length-prefixed records. Appends use
// inter-process file locking and fsync (or FlushFileBuffers on Windows).
// Reads use mmap.
//
// PosixEventSink holds no per-(tenant, session) cache of its own. Appends use
// a process-local path mutex plus an inter-process file lock; callers that
// need a decoded-event cache should layer it above the sink (see
// EventSourcedLog).
//
// The on-disk format is suitable for an S3 Files mount (NFS 4.x via
// mount -t s3files), an EFS mount, or local disk. See
// runtime/dpm/PHASE1_AUDIT_RECOVERY.md for the empirical verification.
class PosixEventSink : public EventSink {
 public:
  explicit PosixEventSink(std::filesystem::path root_path);

  absl::Status AppendRecord(absl::string_view tenant_id,
                            absl::string_view session_id,
                            absl::string_view record_payload) override;

  absl::StatusOr<std::vector<std::string>> ReadRecords(
      absl::string_view tenant_id,
      absl::string_view session_id) const override;

  absl::Status ForEachRecord(
      absl::string_view tenant_id, absl::string_view session_id,
      absl::FunctionRef<absl::Status(absl::string_view)> callback)
      const override;

  absl::StatusOr<EventSink::Generation> ProbeGeneration(
      absl::string_view tenant_id,
      absl::string_view session_id) const override;

  std::filesystem::path PathFor(absl::string_view tenant_id,
                                absl::string_view session_id) const;

 private:
  std::filesystem::path root_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_EVENTLOG_POSIX_EVENT_SINK_H_
