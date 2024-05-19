cmake_minimum_required(VERSION 3.10)

# Projekt neve
project(Algoritmusok)

# Output directory
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)

# Input és output fájlok
set(MARKDOWN_FILE ${CMAKE_SOURCE_DIR}/computer_vision.md)
set(PDF_FILE ${OUTPUT_DIR}/computer_vision.pdf)

# Kimeneti könyvtár létrehozása
file(MAKE_DIRECTORY ${OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(common)
# Pandoc parancs hozzáadása egy custom target-hez
#add_book_target(${MARKDOWN_FILE} ${PDF_FILE})

add_subdirectory(sources)
