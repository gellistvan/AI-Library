set(PART_1_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Templates/1_Template_metaprogramming.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Templates/2_TMP_and_Modern_C++.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Templates/3_Expression_templates.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Templates/4_Curiously_Recurring_Template_Pattern.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_1_Templates/5_Type_Erasure_and_Polymorphism.md")

set(PART_2_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_STL/6_Custom_and_Extended_STL_Containers.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_STL/7_Iterator_mastery.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_STL/8_Algorithmic_Techniques.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_2_STL/9_Advanced_Usage_of_Specific_Containers.md")

set(PART_3_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Memory/10_Memory_Management_Techniques.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Memory/11_Interfacing_with_Low-Level_Memory.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Memory/12_Memory_Models_and_Concurrency.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Memory/13_Efficient_String_and_Buffer_Management.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_3_Memory/14_Optimizations_and_Performance_Tuning.md")

set(PART_4_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Preprocessor/15_Advanced_Macro_Techniques_and_Metaprogramming.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Preprocessor/16_Stringizing,_Concatenation,_and_Code_Generation.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Preprocessor/17_Using_the_Preprocessor_for_Documentation.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_4_Preprocessor/18_Pragmas_and_Compiler-Specific_Extensions.md")

set(PART_5_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/19_Structural_Patterns.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/20_Behavioral_Patterns.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/21_Concurrency_Patterns.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/22_Event-Driven_and_Reactive_Patterns.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/23_Architectural_Patterns.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_5_Design_Patterms/24_Modern_C++_Idioms_and_Best_Practices.md")

set(PART_6_CHAPTERS
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_Everything_else/25_Reflection_and_Introspection.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_Everything_else/26_Domain-Specific_Languages.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/Part_6_Everything_else/27_Interfacing_with_Other_Languages.md")

set(ALL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/Introduction.md"
        ${PART_1_CHAPTERS}
        ${PART_2_CHAPTERS}
        ${PART_3_CHAPTERS}
        ${PART_4_CHAPTERS}
        ${PART_5_CHAPTERS}
        ${PART_6_CHAPTERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/Closure.md"
)

add_combined_book_target(advanced_cpp_part_1 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_1_CHAPTERS}")
add_combined_book_target(advanced_cpp_part_2 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_2_CHAPTERS}")
add_combined_book_target(advanced_cpp_part_3 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_3_CHAPTERS}")
add_combined_book_target(advanced_cpp_part_4 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_4_CHAPTERS}")
add_combined_book_target(advanced_cpp_part_5 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_5_CHAPTERS}")
add_combined_book_target(advanced_cpp_part_6 ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${PART_6_CHAPTERS}")
add_combined_book_target(advanced_cpp ${CMAKE_CURRENT_SOURCE_DIR}/preamble_a4.yml a4 "${ALL_SOURCE_FILES}")

