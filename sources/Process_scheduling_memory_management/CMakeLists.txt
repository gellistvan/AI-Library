set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/1_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_5.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_6.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_5.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_6.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_7.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_8.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_3.md"
)


foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble_a4.yml a4.pdf)
endforeach()

add_combined_book_target(process_scheduling ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")
