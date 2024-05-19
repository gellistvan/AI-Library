macro(add_book_target MARKDOWN_FILE PDF_FILE)
    # Get the directory of the markdown file relative to the source directory
    get_filename_component(MARKDOWN_DIR ${MARKDOWN_FILE} DIRECTORY)

    # Get the relative path from the source directory to the markdown file directory
    file(RELATIVE_PATH REL_DIR ${CMAKE_SOURCE_DIR} ${MARKDOWN_DIR})

    # Construct the output directory based on the relative path
    set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/output/${REL_DIR})

    # Ensure the output directory exists
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # Get the filename without the directory
    get_filename_component(FILE_NAME ${PDF_FILE} NAME)

    # Construct the full path for the PDF file in the output directory
    set(OUTPUT_PDF ${OUTPUT_DIR}/${FILE_NAME})

    # Add custom target to generate PDF
    add_custom_target(
            ${PDF_FILE}
            ALL
            COMMAND pandoc ${MARKDOWN_FILE} -o ${OUTPUT_PDF}
            DEPENDS ${MARKDOWN_FILE}
            COMMENT "Generating PDF from Markdown using Pandoc"
            WORKING_DIRECTORY ${MARKDOWN_DIR}
    )

    message(STATUS "Generating PDF from ${MARKDOWN_FILE} to ${OUTPUT_PDF}")
endmacro()
