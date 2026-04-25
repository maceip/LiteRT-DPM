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

add_litertlm_library(runtime_dpm_config_dpm_config_loader STATIC
  dpm_config_loader.cc
)
add_library(LiteRTLM::Runtime::DPM::Config::DpmConfigLoader ALIAS
  runtime_dpm_config_dpm_config_loader)

target_include_directories(runtime_dpm_config_dpm_config_loader
  PUBLIC
    ${GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_dpm_config_dpm_config_loader
  PUBLIC
    runtime_dpm_dpm
    runtime_proto_dpm_config_cc_proto
    LITERTLM_DEPS
)
