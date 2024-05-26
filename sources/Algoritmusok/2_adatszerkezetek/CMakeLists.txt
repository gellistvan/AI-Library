include(common)

set(PART_1_CHAPTERS
        "../0_eloszo.md"
        "../1_1_beezetes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_alapveto_adatszerkezetek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_hash_tablak.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_fa_szerkezetek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_heap.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_graf_adatszerkezetek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/6_trie.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/7_fenwick_fa.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/8_segment_fa.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/9_sufix_fa_array.md"
)

add_combined_book_target(adatszerkezetek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${PART_1_CHAPTERS}")

