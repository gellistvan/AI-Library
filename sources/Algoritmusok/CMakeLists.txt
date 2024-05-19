include(common)

set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/adv.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/adv2.md"
        # Add other chapters as needed
)

foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble.yml a4.pdf)
endforeach()

add_combined_book_target(algoritmusok_combined ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${CHAPTERS}")

