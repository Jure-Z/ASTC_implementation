cmake_minimum_required(VERSION 3.0...3.25)
project(
    webgpu_astc
    VERSION 0.1.0
    LANGUAGES CXX C
)

option(DEV_MODE "Set up development helper settings" ON)

# This function automates the process of embedding a file into a C++ header
function(embed_file target_name input_file_absolute namespace var_name)
    set(output_header "${CMAKE_CURRENT_BINARY_DIR}/generated/${var_name}.h")

    # This command tells CMake how to generate the header from the input file
    add_custom_command(
        OUTPUT ${output_header}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/embed.py
                ${input_file_absolute}
                ${output_header}
                ${namespace}
                ${var_name}
        DEPENDS ${input_file_absolute}
                ${CMAKE_CURRENT_SOURCE_DIR}/embed.py
        COMMENT "Embedding ${input_file_absolute} into ${output_header}"
    )

    # This creates a target that represents the generated file
    add_custom_target(
        embed_${var_name}
        DEPENDS ${output_header}
    )

    # Add the directory of the generated file to the include path for our main target
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

    # Make our main target depend on the generation of this file
    add_dependencies(${target_name} embed_${var_name})
endfunction()

# === Source files in /code ===
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS code/*.cpp)

# === Source files ===
add_executable(webgpu_astc ${SOURCES})

#embed the shader files for native build
if(NOT EMSCRIPTEN)
    set(GENERATED_SHADER_HEADERS "")

    file(GLOB SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.wgsl")

    foreach(shader_file_abs ${SHADER_FILES})
        # Create a clean variable name from the full path
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" relative_path ${shader_file_abs})
        string(REPLACE "/" "_" var_name_temp ${relative_path})
        string(REPLACE "." "_" var_name ${var_name_temp})
    
        embed_file(webgpu_astc ${shader_file_abs} Shaders ${var_name})

        list(APPEND GENERATED_SHADER_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/generated/${var_name}.h")
    endforeach()

    target_sources(webgpu_astc PRIVATE ${GENERATED_SHADER_HEADERS})

endif()

set_target_properties(webgpu_astc PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON
	CXX_EXTENSIONS OFF
)

if(EMSCRIPTEN)

    message(STATUS "Configuring for WebAssembly with Emscripten")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")

    set(EMSC_LINK_FLAGS
        "-sUSE_WEBGPU=1"
        "-sASYNCIFY=1"
        "-sINITIAL_MEMORY=64MB"
        "-sMAXIMUM_MEMORY=2048MB"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sDISABLE_EXCEPTION_CATCHING=0"
        # "-sLLD_REPORT_UNDEFINED"

        "--preload-file" "${CMAKE_CURRENT_SOURCE_DIR}/shaders@shaders"

        "--shell-file" "${CMAKE_CURRENT_SOURCE_DIR}/shell_template.html"

        "-sEXPORTED_FUNCTIONS=['_malloc','_free','_main']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall', 'cwrap', 'HEAPU8']"
    )

    # Apply the link flags specifically to the target as well, this is often more robust
    target_link_options(webgpu_astc PRIVATE ${EMSC_LINK_FLAGS})
else()

    find_package(Python3 REQUIRED)

endif()

if(DEV_MODE)
    target_compile_definitions(webgpu_astc PRIVATE
        TEST_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_images"
    )
else()
    target_compile_definitions(webgpu_astc PRIVATE
        TEST_IMAGE_DIR="./test_images"
    )
endif()

add_subdirectory(webgpu)

target_link_libraries(webgpu_astc PRIVATE webgpu)

# === Native specific post-build step (Dawn binaries) ===
if(NOT EMSCRIPTEN)
    # This is likely from webgpu-dawn, and only for native
    # If target_copy_webgpu_binaries is not defined for Emscripten, this is fine.
    # Ensure webgpu/CMakeLists.txt defines this conditionally.
    if(TARGET target_copy_webgpu_binaries) # Check if the function/macro exists
        target_copy_webgpu_binaries(webgpu_astc)
    else()
        message(WARNING "target_copy_webgpu_binaries macro not found. Skipping.")
    endif()
endif()