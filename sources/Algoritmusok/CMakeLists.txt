add_subdirectory(2_adatszerkezetek)
add_subdirectory(3_Algoritmusok)

set(ALGORITHMS
        ${1_SEARCH}
        ${2_ORDER}
        ${3_DYNAMIC}
        ${4_GREEDY}
        ${5_BACKTRACK}
        ${6_DIVIDE}
        ${7_HALOZATI}
)

set(ALL_CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/0_eloszo.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_1_beezetes.md"
        ${PART_2_CHAPTERS}
        ${ALGORITHMS}
)


#add_combined_book_target(3_1_keresesek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${1_SEARCH}")

#add_combined_book_target(3_2_rendezesek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${2_ORDER}")
#add_combined_book_target(3_3_dinamikus ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${3_DYNAMIC}")

#add_combined_book_target(3_4_moho ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${4_GREEDY}")
#add_combined_book_target(3_5_backtrack ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${5_BACKTRACK}")

#add_combined_book_target(3_6_oszd_meg ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${6_DIVIDE}")
#add_combined_book_target(3_7_halozati ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${7_HALOZATI}")

add_combined_book_target(adatszerkezetek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${PART_2_CHAPTERS}")
add_combined_book_target(algoritmusok ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${ALGORITHMS}")
#add_combined_book_target(algoritmusok_es_adatszerkezetek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${ALL_CHAPTERS}")
