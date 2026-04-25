# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(PKG_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_litertlm_library(runtime_platform_eventlog_posix_event_sink STATIC
  posix_event_sink.cc
)
add_library(LiteRTLM::Runtime::Platform::Eventlog::PosixEventSink ALIAS
  runtime_platform_eventlog_posix_event_sink)

target_include_directories(runtime_platform_eventlog_posix_event_sink
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_platform_eventlog_posix_event_sink
  PUBLIC
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

add_library(runtime_platform_eventlog_libs INTERFACE)
add_library(LiteRTLM::Runtime::Platform::Eventlog ALIAS
  runtime_platform_eventlog_libs)

target_link_libraries(runtime_platform_eventlog_libs INTERFACE
  LiteRTLM::Runtime::Platform::Eventlog::PosixEventSink
)
