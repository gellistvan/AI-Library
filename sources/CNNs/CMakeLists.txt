set(CHAPTERS        
        "${CMAKE_CURRENT_SOURCE_DIR}/1_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_3.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_5.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_6.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_7.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_8.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_9.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_10.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_11.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_12.md"
)


foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble_a4.yml a4.pdf)
endforeach()

add_combined_book_target(cnn ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")
