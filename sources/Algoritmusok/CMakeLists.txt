include(common)

# List of chapters
set(CHAPTERS
        "adv.md"
        "adv2.md"
        # Add other chapters as needed
)

# Loop through the list of chapters and create targets
foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CMAKE_CURRENT_SOURCE_DIR}/${CHAPTER} ${CHAPTER_NAME}.pdf)
endforeach()
