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

#include "runtime/platform/eventlog/posix_event_sink.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "gtest/gtest.h"
#include "runtime/util/test_utils.h"

namespace litert::lm {
namespace {

std::filesystem::path TestRoot(absl::string_view name) {
  std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / std::string(name);
  std::filesystem::remove_all(path);
  return path;
}

TEST(PosixEventSinkTest, AppendsAndReadsBytesInOrder) {
  PosixEventSink sink(TestRoot("posix_event_sink_basic"));
  ASSERT_OK(sink.AppendRecord("tenant-a", "session-1", "first"));
  ASSERT_OK(sink.AppendRecord("tenant-a", "session-1", "second"));

  ASSERT_OK_AND_ASSIGN(std::vector<std::string> records,
                       sink.ReadRecords("tenant-a", "session-1"));
  ASSERT_EQ(records.size(), 2);
  EXPECT_EQ(records[0], "first");
  EXPECT_EQ(records[1], "second");
}

TEST(PosixEventSinkTest, IsolatesTenantsAndSessionsByPath) {
  PosixEventSink sink(TestRoot("posix_event_sink_isolation"));
  ASSERT_OK(sink.AppendRecord("tenant-a", "session-1", "AAA"));
  ASSERT_OK(sink.AppendRecord("tenant-b", "session-1", "BBB"));
  ASSERT_OK(sink.AppendRecord("tenant-a", "session-2", "CCC"));

  ASSERT_OK_AND_ASSIGN(auto a1, sink.ReadRecords("tenant-a", "session-1"));
  ASSERT_OK_AND_ASSIGN(auto b1, sink.ReadRecords("tenant-b", "session-1"));
  ASSERT_OK_AND_ASSIGN(auto a2, sink.ReadRecords("tenant-a", "session-2"));
  ASSERT_EQ(a1.size(), 1);
  EXPECT_EQ(a1[0], "AAA");
  ASSERT_EQ(b1.size(), 1);
  EXPECT_EQ(b1[0], "BBB");
  ASSERT_EQ(a2.size(), 1);
  EXPECT_EQ(a2[0], "CCC");
}

TEST(PosixEventSinkTest, RejectsPathTraversalIdentities) {
  PosixEventSink sink(TestRoot("posix_event_sink_traversal"));
  EXPECT_FALSE(sink.AppendRecord("..", "session-1", "x").ok());
  EXPECT_FALSE(sink.AppendRecord("tenant-a", "..", "x").ok());
  EXPECT_FALSE(sink.AppendRecord("ten/ant", "session-1", "x").ok());
  EXPECT_FALSE(sink.AppendRecord("", "session-1", "x").ok());
}

TEST(PosixEventSinkTest, ReturnsEmptyForUnknownSession) {
  PosixEventSink sink(TestRoot("posix_event_sink_empty"));
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> records,
                       sink.ReadRecords("tenant-a", "session-1"));
  EXPECT_TRUE(records.empty());
}

TEST(PosixEventSinkTest, DetectsCorruptedTrailingByte) {
  PosixEventSink sink(TestRoot("posix_event_sink_partial"));
  ASSERT_OK(sink.AppendRecord("tenant-a", "session-1", "first"));
  std::ofstream file(sink.PathFor("tenant-a", "session-1"),
                     std::ios::out | std::ios::app | std::ios::binary);
  file.put('\1');
  file.close();
  EXPECT_FALSE(sink.ReadRecords("tenant-a", "session-1").ok());
}

}  // namespace
}  // namespace litert::lm
