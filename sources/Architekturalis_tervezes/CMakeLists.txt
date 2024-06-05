set(CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/02_az architektura.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/03_szofvertervezes_alapjai.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/04_szoftverarchitektura_elvek.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/05_nfr.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/06_kovetelmenyanalizis.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/07_architektura_tervezes.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/08_modellezesi_eszkozok.md"
#    "${CMAKE_CURRENT_SOURCE_DIR}/06_architektura_tervezes.md"

#    "${CMAKE_CURRENT_SOURCE_DIR}/07_modellezesi_eszkozok.md"
#    "${CMAKE_CURRENT_SOURCE_DIR}/08_ui_ux_tervezes.md"

#    "${CMAKE_CURRENT_SOURCE_DIR}/09_biztonsag.md"
#    "${CMAKE_CURRENT_SOURCE_DIR}/10_performancia.md"

#    "${CMAKE_CURRENT_SOURCE_DIR}/11_refactoring.md"
#    "${CMAKE_CURRENT_SOURCE_DIR}/12_esettanulmanyok.md"

#    "1"
#    "1"

#    "1"
#    "1"

#    "1"
#    "1"

#    "1"
#    "1"

#    "1"
#    "1"
)

add_combined_book_target(architectural_design ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")

