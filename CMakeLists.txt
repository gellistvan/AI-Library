cmake_minimum_required(VERSION 3.10)

project(AI_Library)

set(CMAKE_INSTALL_BINARY_DIR ${CMAKE_SOURCE_DIR}/output)
message(STATUS "BIN_DIR:${CMAKE_INSTALL_BINARY_DIR}")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(common)

add_subdirectory(sources)
