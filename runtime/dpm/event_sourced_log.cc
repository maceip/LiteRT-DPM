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

#include "runtime/dpm/event_sourced_log.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/event.h"
#include "runtime/platform/eventlog/event_sink.h"
#include "runtime/platform/eventlog/posix_event_sink.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

bool IsValidIdentityComponent(absl::string_view value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find('/') == absl::string_view::npos &&
         value.find('\\') == absl::string_view::npos;
}

}  // namespace

EventSourcedLog::EventSourcedLog(std::filesystem::path root_path,
                                 DPMLogIdentity identity)
    : owned_sink_(std::make_unique<PosixEventSink>(std::move(root_path))),
      sink_(owned_sink_.get()),
      identity_(std::move(identity)) {}

EventSourcedLog::EventSourcedLog(EventSink* sink, DPMLogIdentity identity)
    : owned_sink_(nullptr), sink_(sink), identity_(std::move(identity)) {}

std::filesystem::path EventSourcedLog::path() const {
  if (auto* posix = dynamic_cast<PosixEventSink*>(sink_)) {
    return posix->PathFor(identity_.tenant_id, identity_.session_id);
  }
  return {};
}

absl::Status EventSourcedLog::ValidateIdentity() const {
  if (!IsValidIdentityComponent(identity_.tenant_id)) {
    return absl::InvalidArgumentError(
        "DPM tenant_id must be non-empty and must not contain path separators.");
  }
  if (!IsValidIdentityComponent(identity_.session_id)) {
    return absl::InvalidArgumentError(
        "DPM session_id must be non-empty and must not contain path "
        "separators.");
  }
  return absl::OkStatus();
}

absl::Status EventSourcedLog::ValidateEventIdentity(const Event& event) const {
  if (event.tenant_id != identity_.tenant_id ||
      event.session_id != identity_.session_id) {
    return absl::InvalidArgumentError(
        "DPM event identity does not match the owning EventSourcedLog.");
  }
  return absl::OkStatus();
}

absl::Status EventSourcedLog::Append(Event event) {
  if (sink_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM EventSourcedLog has no event sink bound.");
  }
  RETURN_IF_ERROR(ValidateIdentity());
  if (event.tenant_id.empty()) {
    event.tenant_id = identity_.tenant_id;
  }
  if (event.session_id.empty()) {
    event.session_id = identity_.session_id;
  }
  RETURN_IF_ERROR(ValidateEventIdentity(event));

  std::lock_guard<std::mutex> lock(mutex_);
  const std::string payload = EventToJsonLine(event);
  RETURN_IF_ERROR(sink_->AppendRecord(identity_.tenant_id,
                                      identity_.session_id, payload));
  cache_loaded_ = false;
  cached_record_count_ = 0;
  cached_events_.clear();
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Event>> EventSourcedLog::LoadEventsLocked() const {
  ASSIGN_OR_RETURN(std::vector<std::string> records,
                   sink_->ReadRecords(identity_.tenant_id,
                                      identity_.session_id));
  if (cache_loaded_ && cached_record_count_ == records.size()) {
    return cached_events_;
  }
  std::vector<Event> events;
  events.reserve(records.size());
  for (size_t i = 0; i < records.size(); ++i) {
    absl::StatusOr<Event> event = EventFromJsonLine(records[i]);
    if (!event.ok()) {
      return absl::DataLossError(
          absl::StrCat("DPM event log record ", i,
                       " failed to parse: ", event.status().message()));
    }
    if (event->tenant_id != identity_.tenant_id ||
        event->session_id != identity_.session_id) {
      return absl::DataLossError(
          "DPM event log contains a cross-tenant or cross-session event.");
    }
    events.push_back(std::move(*event));
  }
  cached_events_ = events;
  cached_record_count_ = records.size();
  cache_loaded_ = true;
  return cached_events_;
}

absl::StatusOr<std::vector<Event>> EventSourcedLog::GetAllEvents() const {
  if (sink_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM EventSourcedLog has no event sink bound.");
  }
  RETURN_IF_ERROR(ValidateIdentity());
  std::lock_guard<std::mutex> lock(mutex_);
  return LoadEventsLocked();
}

absl::StatusOr<std::vector<Event>> EventSourcedLog::GetEventsSince(
    absl::string_view checkpoint) const {
  int64_t checkpoint_index = 0;
  try {
    std::string checkpoint_string(checkpoint);
    size_t parsed_chars = 0;
    checkpoint_index = std::stoll(checkpoint_string, &parsed_chars);
    if (parsed_chars != checkpoint_string.size() || checkpoint_index < 0) {
      return absl::InvalidArgumentError(
          "DPM checkpoint must be a non-negative event index.");
    }
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(
        "DPM checkpoint must be a non-negative event index.");
  }

  ASSIGN_OR_RETURN(std::vector<Event> events, GetAllEvents());
  const size_t checkpoint_offset = static_cast<size_t>(checkpoint_index);
  if (checkpoint_offset >= events.size()) {
    return std::vector<Event>();
  }
  return std::vector<Event>(events.begin() + checkpoint_offset, events.end());
}

}  // namespace litert::lm
