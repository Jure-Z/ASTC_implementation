cmake_minimum_required(VERSION 3.0...3.25)
project(
    webgpu_astc
    VERSION 0.1.0
    LANGUAGES CXX C
)

option(DEV_MODE "Set up development helper settings" ON)

# === Source files in /code ===
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS code/*.cpp)

# === Source files ===
add_executable(webgpu_astc ${SOURCES} "code/stb_image.h" "code/webgpu_utils.cpp" "code/webgpu_utils.h" "code/compression_pipeline.cpp"   "code/metadata_structures.cpp" "code/astc.h" "code/physical_compression.cpp" "code/partition_tables.cpp" "code/best_partitionings.cpp" "code/averages_and_directions.cpp" "code/averages_and_directions.h" "code/shaders_init.cpp" "code/astc_store.cpp" "code/astc_store.h")

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
endif()

if(DEV_MODE)
    target_compile_definitions(webgpu_astc PRIVATE
        SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        TEST_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_images"
    )
else()
    target_compile_definitions(webgpu_astc PRIVATE
        SHADER_DIR="./shaders"
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