cmake_minimum_required(VERSION 3.16)
project(facex_frvt_11 CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NIST evaluation expects exactly this naming:
# libfrvt_11_<provider>_<sequence>.so
set(LIB_NAME "frvt_11_facex_001")

if (NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-X.Y.Z")
endif()

include_directories(
    include
    ${ONNXRUNTIME_ROOT}/include
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

add_library(${LIB_NAME} SHARED
    src/facex_frvt.cpp
)

target_compile_options(${LIB_NAME} PRIVATE
    -O3 -march=x86-64-v3 -funroll-loops -Wall -Wextra -Wno-unused-parameter)

target_link_libraries(${LIB_NAME} PRIVATE
    onnxruntime
    pthread
    dl
)

set_target_properties(${LIB_NAME} PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "${LIB_NAME}")

# Install into ./install/lib so we can copy to NIST submission package
install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION lib)
