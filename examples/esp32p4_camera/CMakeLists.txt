# ESP-IDF top-level project for the ESP32-P4 MIPI-CSI camera + FaceX example.
# Build:  idf.py set-target esp32p4
#         idf.py build flash monitor

cmake_minimum_required(VERSION 3.16)

# Make our /components/facex visible to IDF's component search.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../components")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(facex_esp32p4_camera)
