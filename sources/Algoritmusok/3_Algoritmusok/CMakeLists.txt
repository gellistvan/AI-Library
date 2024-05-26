include(common)
set(1_SEARCH
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_1_linearis.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_2_binaris.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_3_interpolacios.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_4_exponencialis.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_5_ugrasos.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_6_kettos_ugrasos.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_7_ternary.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_8_hash_alapu.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_9_string_keresesi_algoritmusok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_10_meta_kereses.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_11_adatbazis_kereses.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_12_parhuzamos_es_elosztott.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_13_helyfuggo_kereses.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_keresesek/1_14_evolucios-es-heurisztikus.md"
)

set(2_ORDER
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_1_buborek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_2_kivalasztasos.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_3_beszuro.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_4_gyorsrendezes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_5_gyorsrendezes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_6_kupac.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_7_shell_sort.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_8_szamlalo_Redezes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_9_radix.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_10_bucket_sort.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_11_comb_sort.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_12_tim_sort.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_13_parhuzamos_es_elosztott.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_14_adaptiv_rendezes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_15_stabil_es_instabil_algoritmusok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_rendezesek/2_16_osszehasonlitas.md"
)

set(3_DYNAMIC
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_1_alapelvek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_3_hatizsak.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_4_kozos_reszsorozat.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_5_novekvo_reszsorozat.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_6_matrix_lanc_szorzas.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_7_szerkesztesi_tavolsag.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_8_ermek_osszegzese.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_dinamikus_programozas/3_9_intervallum_scheduling.md"
)
add_combined_book_target(3_1_keresesek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${1_SEARCH}")
add_combined_book_target(3_2_rendezesek ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${2_ORDER}")
add_combined_book_target(3_3_dinamikus ${CMAKE_CURRENT_SOURCE_DIR}/preamble.yml a4 "${3_DYNAMIC}")