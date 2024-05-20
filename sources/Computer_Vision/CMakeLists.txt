include(common)

set(PART_1_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Introduction/Chapter_1.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Introduction/Chapter_2.md")

set(PART_2_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Image_Processing_Fundamentals/Chapter_3.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Image_Processing_Fundamentals/Chapter_4.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_Image_Processing_Fundamentals/Chapter_5.md")

set(PART_3_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Feature_Detection_and_Matching/Chapter_6.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Feature_Detection_and_Matching/Chapter_7.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Feature_Detection_and_Matching/Chapter_8.md")

set(PART_4_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Image_Segmentation/Chapter_10.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Image_Segmentation/Chapter_11.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Image_Segmentation/Chapter_12.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Image_Segmentation/Chapter_9.md")

set(PART_5_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Object_Detection_and_Recognition/Chapter_13.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Object_Detection_and_Recognition/Chapter_14.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Object_Detection_and_Recognition/Chapter_15.md")

set(PART_6_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_3D_Vision_and_Geometry/Chapter_16.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_3D_Vision_and_Geometry/Chapter_17.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_3D_Vision_and_Geometry/Chapter_18.md")

set(PART_7_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_7_Motion_Analysis/Chapter_19.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_7_Motion_Analysis/Chapter_20.md")

set(PART_8_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_8_Advanced_Topics_and_Applications/Chapter_21.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_8_Advanced_Topics_and_Applications/Chapter_22.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_8_Advanced_Topics_and_Applications/Chapter_23.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_8_Advanced_Topics_and_Applications/Chapter_24.md")

set(PART_9_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_9_Future_Directions/Chapter_25.md")

add_book_target(${CMAKE_CURRENT_SOURCE_DIR}/computer_vision.md preamble.yml a4.pdf)

set(ALL_SOURCE_FILES
    PART_1_CHAPTERS
    PART_2_CHAPTERS
    PART_3_CHAPTERS
    PART_4_CHAPTERS
    PART_5_CHAPTERS
    PART_6_CHAPTERS
    PART_7_CHAPTERS
    PART_8_CHAPTERS
    PART_9_CHAPTERS
)

add_combined_book_target(computer_vision_part_1 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_1_CHAPTERS}")
add_combined_book_target(computer_vision_part_2 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_2_CHAPTERS}")
add_combined_book_target(computer_vision_part_3 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_3_CHAPTERS}")
add_combined_book_target(computer_vision_part_4 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_4_CHAPTERS}")
add_combined_book_target(computer_vision_part_5 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_5_CHAPTERS}")
add_combined_book_target(computer_vision_part_6 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_6_CHAPTERS}")
add_combined_book_target(computer_vision_part_7 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_7_CHAPTERS}")
add_combined_book_target(computer_vision_part_8 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_8_CHAPTERS}")
add_combined_book_target(computer_vision_part_9 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_9_CHAPTERS}")
add_combined_book_target(computer_vision ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${ALL_SOURCE_FILES}")