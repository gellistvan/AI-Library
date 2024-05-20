set(PART_1_CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Embedded_Systems/Chapter_1.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Embedded_Systems/Chapter_2.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Embedded_Systems/Chapter_3.md")


set(PART_2_CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_4.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_5.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_6.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_7.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_8.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_9.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_10.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_11.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_12.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_13.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Practical_tecniques/Chapter_14.md"
)

set(ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/Preface.md"
        ${PART_1_CHAPTERS}
        ${PART_2_CHAPTERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/Closure.md"
)

if(BUILD_BOOK_PARTS)
    add_combined_book_target(embedded_cpp_part_1 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_1_CHAPTERS}")
    add_combined_book_target(embedded_cpp_part_2 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_2_CHAPTERS}")
endif()
add_combined_book_target(embedded_cpp ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${ALL_SOURCE_FILES}")

