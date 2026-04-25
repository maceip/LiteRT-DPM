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

add_litertlm_library(runtime_dpm_dpm STATIC
  dpm_projector.cc
  engine_inference_runner.cc
  event.cc
  event_sourced_log.cc
  projection_prompt.cc
  stateless_decision_engine.cc
)
add_library(LiteRTLM::Runtime::DPM::DPM ALIAS runtime_dpm_dpm)

target_include_directories(runtime_dpm_dpm
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
    ${THIRD_PARTY_DIR}/json/include
)

target_link_libraries(runtime_dpm_dpm
  PUBLIC
    runtime_engine_engine_interface
    runtime_engine_engine_settings
    runtime_engine_io_types
    runtime_platform_eventlog_posix_event_sink
    runtime_util_litert_status_util
    LITERTLM_DEPS
)

add_library(runtime_dpm_libs INTERFACE)
add_library(LiteRTLM::Runtime::DPM ALIAS runtime_dpm_libs)

target_link_libraries(runtime_dpm_libs INTERFACE
  LiteRTLM::Runtime::DPM::DPM
)
