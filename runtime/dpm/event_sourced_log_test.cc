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

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "gtest/gtest.h"
#include "runtime/dpm/event.h"
#include "runtime/platform/eventlog/event_sink.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

std::filesystem::path TestPath(absl::string_view name) {
  std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / std::string(name);
  std::filesystem::remove_all(path);
  return path;
}

TEST(EventSourcedLogTest, AppendsAndReadsEventsInOrder) {
  EventSourcedLog log(TestPath("dpm_event_sourced_log_test"),
                      DPMLogIdentity{
                          .tenant_id = "tenant-a",
                          .session_id = "session-1",
                      });

  ASSERT_OK(log.Append(Event{
      .type = Event::Type::kUser,
      .payload = "first",
      .timestamp_us = 100,
  }));
  ASSERT_OK(log.Append(Event{
      .type = Event::Type::kCorrection,
      .payload = "second",
      .timestamp_us = 200,
  }));

  ASSERT_OK_AND_ASSIGN(std::vector<Event> events, log.GetAllEvents());
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].type, Event::Type::kUser);
  EXPECT_EQ(events[0].tenant_id, "tenant-a");
  EXPECT_EQ(events[0].session_id, "session-1");
  EXPECT_EQ(events[0].payload, "first");
  EXPECT_EQ(events[0].timestamp_us, 100);
  EXPECT_EQ(events[1].type, Event::Type::kCorrection);
  EXPECT_EQ(events[1].payload, "second");

  ASSERT_OK_AND_ASSIGN(std::vector<Event> events_since,
                       log.GetEventsSince("1"));
  ASSERT_EQ(events_since.size(), 1);
  EXPECT_EQ(events_since[0].payload, "second");
}

TEST(EventSourcedLogTest, RejectsInvalidCheckpoint) {
  EventSourcedLog log(TestPath("dpm_event_sourced_log_checkpoint_test"),
                      DPMLogIdentity{
                          .tenant_id = "tenant-a",
                          .session_id = "session-1",
                      });
  EXPECT_FALSE(log.GetEventsSince("not-an-index").ok());
  EXPECT_FALSE(log.GetEventsSince("-1").ok());
}

TEST(EventSourcedLogTest, RejectsCrossTenantAppend) {
  EventSourcedLog log(TestPath("dpm_event_sourced_log_identity_test"),
                      DPMLogIdentity{
                          .tenant_id = "tenant-a",
                          .session_id = "session-1",
                      });
  EXPECT_FALSE(log.Append(Event{
      .type = Event::Type::kUser,
      .tenant_id = "tenant-b",
      .session_id = "session-1",
      .payload = "wrong tenant",
      .timestamp_us = 100,
  }).ok());
}

TEST(EventSourcedLogTest, DetectsPartialRecord) {
  EventSourcedLog log(TestPath("dpm_event_sourced_log_partial_test"),
                      DPMLogIdentity{
                          .tenant_id = "tenant-a",
                          .session_id = "session-1",
                      });
  ASSERT_OK(log.Append(Event{
      .type = Event::Type::kUser,
      .payload = "first",
      .timestamp_us = 100,
  }));
  std::ofstream file(log.path(), std::ios::out | std::ios::app |
                                     std::ios::binary);
  file.put('\1');
  file.close();

  EXPECT_FALSE(log.GetAllEvents().ok());
}

// In-memory EventSink fake exercises the substrate-injected constructor and
// confirms that EventSourcedLog is decoupled from the on-disk format.
class InMemoryEventSink : public EventSink {
 public:
  absl::Status AppendRecord(absl::string_view tenant_id,
                            absl::string_view session_id,
                            absl::string_view record_payload) override {
    auto& bucket = records_[std::string(tenant_id) + "|" +
                            std::string(session_id)];
    bucket.emplace_back(record_payload);
    return absl::OkStatus();
  }
  absl::StatusOr<std::vector<std::string>> ReadRecords(
      absl::string_view tenant_id,
      absl::string_view session_id) const override {
    const std::string key =
        std::string(tenant_id) + "|" + std::string(session_id);
    auto it = records_.find(key);
    if (it == records_.end()) {
      return std::vector<std::string>();
    }
    return it->second;
  }

 private:
  mutable std::map<std::string, std::vector<std::string>> records_;
};

TEST(EventSourcedLogTest, RoundTripsThroughInjectedSink) {
  InMemoryEventSink sink;
  EventSourcedLog log(&sink, DPMLogIdentity{
                                 .tenant_id = "tenant-a",
                                 .session_id = "session-1",
                             });
  ASSERT_OK(log.Append(Event{
      .type = Event::Type::kUser,
      .payload = "first",
      .timestamp_us = 100,
  }));
  ASSERT_OK(log.Append(Event{
      .type = Event::Type::kModel,
      .payload = "second",
      .timestamp_us = 200,
  }));

  ASSERT_OK_AND_ASSIGN(std::vector<Event> events, log.GetAllEvents());
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].payload, "first");
  EXPECT_EQ(events[0].tenant_id, "tenant-a");
  EXPECT_EQ(events[1].type, Event::Type::kModel);
  EXPECT_EQ(events[1].timestamp_us, 200);
}

}  // namespace
}  // namespace litert::lm
