set(CHAPTERS
        "${CMAKE_CURRENT_SOURCE_DIR}/00_Eloszo.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/1_bevezetes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/2_alapfogalmak_es_attekintes.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/3_programozasi_nyelvek.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/4_bevezetes_a_cudaba.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/5_cuda_architektura.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/6_fejlesztokornyezet.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/7_cuda_alapok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/8_memoria_kezeles.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/9_halado_cuda.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/10_teljesitmenyoptimalizalas.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/11_konyvtarak_es_apik.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/12_valos_alkalmazasok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/13_cuda_jovoje.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/14_opencl_kornyezet.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/15_opencl_alapok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/16_opencl_memoriakezeles.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/17_halado_opencl.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/18_opencl_optimalizatio.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/19_opencl_alkalmazasok.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/20_hibakereses.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/21_zarszo.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/fuggelek.md"
)


foreach(CHAPTER ${CHAPTERS})
    get_filename_component(CHAPTER_NAME ${CHAPTER} NAME_WE)
    add_book_target(${CHAPTER} preamble_a4.yml a4.pdf)
endforeach()

add_combined_book_target(gpgpu ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${CHAPTERS}")
