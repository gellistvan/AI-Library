macro(add_book_target MARKDOWN_FILE PREAMBLE_FILE SUFFIX)
    # Get the directory of the markdown file relative to the source directory
    get_filename_component(MARKDOWN_DIR ${MARKDOWN_FILE} DIRECTORY)

    # Get the relative path from the source directory to the markdown file directory
    file(RELATIVE_PATH REL_DIR ${CMAKE_SOURCE_DIR}/sources/ ${MARKDOWN_DIR})

    # Construct the output directory based on the relative path
    set(OUTPUT_DIR ${CMAKE_INSTALL_BINARY_DIR}/${REL_DIR})
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # Get the filename without the directory and add the suffix
    get_filename_component(FILE_NAME ${MARKDOWN_FILE} NAME_WE)

    set(OUTPUT_FILE ${FILE_NAME}_${SUFFIX})
    set(OUTPUT_FILE_PATH ${OUTPUT_DIR}/${FILE_NAME}_${SUFFIX})

    # Add custom target to generate PDF
    add_custom_target(
            ${OUTPUT_FILE}
            ALL
            COMMAND pandoc ${MARKDOWN_FILE} -o ${OUTPUT_FILE_PATH} --metadata-file=${PREAMBLE_FILE}
            DEPENDS ${MARKDOWN_FILE}
            COMMENT "Generating PDF from Markdown using Pandoc with preamble ${PREAMBLE_FILE}"
            WORKING_DIRECTORY ${MARKDOWN_DIR}
    )

    message(STATUS "Generating FILE to ${OUTPUT_FILE_PATH}")
endmacro()


macro(add_combined_book_target TARGET_NAME PREAMBLE_FILE SUFFIX MARKDOWN_FILES)
    # Create a space-separated list of Markdown files
    get_filename_component(MARKDOWN_DIR ${PREAMBLE_FILE} DIRECTORY)
    file(RELATIVE_PATH REL_DIR ${CMAKE_SOURCE_DIR}/sources/ ${MARKDOWN_DIR})

    set(MARKDOWN_FILES_LIST "")
    foreach(MARKDOWN_FILE ${MARKDOWN_FILES})
        set(MARKDOWN_FILES_LIST "${MARKDOWN_FILES_LIST} ${MARKDOWN_FILE}")
    endforeach()

    # Set the output PDF file name
    set(OUTPUT_DIR ${CMAKE_INSTALL_BINARY_DIR}/${REL_DIR})
    set(OUTPUT_PDF ${OUTPUT_DIR}/${TARGET_NAME}_${SUFFIX}.pdf)
    file(MAKE_DIRECTORY ${OUTPUT_DIR})


    # Add custom target to generate combined PDF
    add_custom_target(
            ${TARGET_NAME}_${SUFFIX}
            ALL
            COMMAND pandoc ${MARKDOWN_FILES} -o ${OUTPUT_PDF} --metadata-file=${PREAMBLE_FILE}
            DEPENDS ${MARKDOWN_FILES}
            COMMENT "Generating combined PDF from Markdown using Pandoc with preamble ${PREAMBLE_FILE}"
            WORKING_DIRECTORY ${MARKDOWN_DIR}
    )

    message(STATUS "Generating combined PDF ${OUTPUT_PDF}")
endmacro()