set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/1_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_3.md"

)

add_combined_book_target(process_management_and_memory ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")
