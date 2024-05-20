set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_5.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_6.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_7.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_8.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_9.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Chapter_10.md"
        # Add other chapters as needed
)

foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble_a4.yml a4.pdf)
endforeach()

set(ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/Preface.md"
        ${CHAPTERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/Appendix.md"
)

add_combined_book_target(cache_coherence ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${ALL_SOURCE_FILES}")
