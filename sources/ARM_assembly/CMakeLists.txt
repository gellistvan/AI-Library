set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/01_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/02_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/03_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/04_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/05_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/06_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/07_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/08_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/09_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/10_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/11_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/12_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/13_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/14_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/15_chapter.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/16_closure.md"
)

add_combined_book_target(arm_assembly ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")

