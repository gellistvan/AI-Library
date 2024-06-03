set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/1_Introduction.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_Qt_Core_Basics.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_QWidget.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_QTQuick_and_QML.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_Graphics_and_Multimedia.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/6_network_and_databases.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/7_concurency.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/8_opencv_with_qt.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/9_best_practices_and_testing.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/10_deployment.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/11_model_view_programming.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/12_advanced_topics.md"
)

foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble_a4.yml a4.pdf)
endforeach()

add_combined_book_target(qt_basics_in_cpp ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")
