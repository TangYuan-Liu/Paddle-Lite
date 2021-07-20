# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.apu")

if (LITE_WITH_NNADAPTER)
    set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.nnadapter")
endif()

if (LITE_WITH_OPENCL)
    set(INFER_LITE_PUBLISH_ROOT "${INFER_LITE_PUBLISH_ROOT}.opencl")
endif(LITE_WITH_OPENCL)

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)
    if (NOT LITE_ON_TINY_PUBLISH)
        # add cxx lib
        add_custom_target(publish_inference_cxx_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_full_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/gen_code/paddle_code_generator" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/test_model_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                COMMAND cp "${CMAKE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                )
        add_dependencies(publish_inference_cxx_lib paddle_code_generator)
        add_dependencies(publish_inference_cxx_lib bundle_full_api)
        add_dependencies(publish_inference_cxx_lib bundle_light_api)
        add_dependencies(publish_inference_cxx_lib test_model_bin)
        add_dependencies(publish_inference_cxx_lib benchmark_bin)
        if (ARM_TARGET_OS STREQUAL "android" OR ARM_TARGET_OS STREQUAL "armlinux" OR ARM_TARGET_OS STREQUAL "armmacos")
            add_dependencies(publish_inference_cxx_lib paddle_full_api_shared)
            add_dependencies(publish_inference paddle_light_api_shared)
            add_custom_command(TARGET publish_inference_cxx_lib
                  COMMAND cp ${CMAKE_BINARY_DIR}/lite/api/*.so ${INFER_LITE_PUBLISH_ROOT}/cxx/lib
                  COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/benchmark_bin" "${INFER_LITE_PUBLISH_ROOT}/bin"
                  )
        endif()
        add_dependencies(publish_inference publish_inference_cxx_lib)
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET publish_inference_cxx_lib POST_BUILD
                    COMMAND ${CMAKE_STRIP} "--strip-debug" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/*.a
                    COMMAND ${CMAKE_STRIP} "--strip-debug" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/*.so)
        endif()
    else()
        # compile cplus shared library, pack the cplus demo and lib into the publish directory.
        add_custom_target(tiny_publish_cxx_lib ${TARGET}
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_SOURCE_DIR}/lite/api/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/libpaddle_light_api_shared.so" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
            COMMAND cp "${CMAKE_SOURCE_DIR}/lite/utils/cv/paddle_*.h" "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
            )
        add_dependencies(tiny_publish_cxx_lib paddle_light_api_shared)
        add_dependencies(publish_inference tiny_publish_cxx_lib)
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET tiny_publish_cxx_lib POST_BUILD
                        COMMAND ${CMAKE_STRIP} "-s" ${INFER_LITE_PUBLISH_ROOT}/cxx/lib/libpaddle_light_api_shared.so)
        endif()
            # compile cplus static library, pack static lib into the publish directory.
        if(LITE_WITH_STATIC_LIB)
            add_custom_target(tiny_publish_cxx_static_lib ${TARGET}
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/include"
                COMMAND mkdir -p "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                COMMAND cp "${CMAKE_BINARY_DIR}/lite/api/libpaddle_api_light_bundled.a" "${INFER_LITE_PUBLISH_ROOT}/cxx/lib"
                )
            add_dependencies(tiny_publish_cxx_static_lib paddle_api_light_bundled)
            add_dependencies(publish_inference tiny_publish_cxx_static_lib)
        endif()
endif()