# components/facex — ESP-IDF wrapper around the FaceX C engine.
#
# This is the integration seam between the FaceX library and an
# ESP-IDF application. Three operating modes, selected via Kconfig:
#
#   FACEX_BACKEND_STUB    (default)
#       facex_esp_detect() returns synthetic faces — useful for
#       wiring up the camera + UI without committing to a real
#       neural network. Zero RAM / zero flash beyond the wrapper.
#
#   FACEX_BACKEND_NATIVE  (advanced)
#       Pulls in the existing src/edgeface_engine.c + transformer_ops.
#       Will NOT fit on ESP32-P4 with EdgeFace-XS (1.77M params, ~7MB
#       FP32 / ~1.8MB INT8). Provided for evaluation only — see
#       docs/esp32p4.md for the EdgeFace-Nano + ESP-NN path that
#       actually fits (future work).
#
#   FACEX_BACKEND_ESPNN   (reserved, not yet implemented)
#       Will dispatch convs through esp-nn (PIE-SIMD INT8 on P4).

set(srcs
    "src/facex_esp.c"
)

set(includes
    "include"
    "${CMAKE_CURRENT_LIST_DIR}/../../include"   # for facex.h
)

set(reqs
    esp_timer
    log
    heap
)

if(CONFIG_FACEX_BACKEND_NATIVE)
    list(APPEND srcs
        "${CMAKE_CURRENT_LIST_DIR}/../../src/transformer_ops.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/edgeface_engine.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/gemm_stub.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/threadpool_stub.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/align.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/detect.c"
        "${CMAKE_CURRENT_LIST_DIR}/../../src/facex.c"
    )
    list(APPEND reqs spi_flash esp_psram)
endif()

idf_component_register(
    SRCS         ${srcs}
    INCLUDE_DIRS ${includes}
    REQUIRES     ${reqs}
)

# The native engine assumes pthread / aligned_alloc / mmap — none are
# available on ESP-IDF. We compile with FACEX_NO_INT8 + a stub flag so
# the engine's malloc-based path is taken; PSRAM provides the budget.
if(CONFIG_FACEX_BACKEND_NATIVE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        FACEX_NO_INT8
        FACEX_LIB
        FACEX_TARGET_ESP32P4
    )
endif()
